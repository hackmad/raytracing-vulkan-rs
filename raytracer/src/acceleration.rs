use std::{collections::HashMap, iter, mem::size_of, sync::Arc};

use anyhow::{Context, Result};
use log::{debug, warn};
use shaders::ray_gen::MeshVertex;
use vulkano::{
    Packed24_8,
    acceleration_structure::{
        AccelerationStructure, AccelerationStructureBuildGeometryInfo,
        AccelerationStructureBuildRangeInfo, AccelerationStructureBuildType,
        AccelerationStructureCreateInfo, AccelerationStructureGeometries,
        AccelerationStructureGeometryInstancesData, AccelerationStructureGeometryInstancesDataType,
        AccelerationStructureGeometryTrianglesData, AccelerationStructureInstance,
        AccelerationStructureType, BuildAccelerationStructureFlags, BuildAccelerationStructureMode,
    },
    buffer::{Buffer, BufferCreateInfo, BufferUsage, IndexBuffer, Subbuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBufferAbstract},
    format::Format,
    memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter},
    sync::GpuFuture,
};

use crate::{Mesh, MeshInstance, Vk};

/// Stores the acceleration structures.
pub struct AccelerationStructures {
    /// The top-level acceleration structure.
    pub tlas: Arc<AccelerationStructure>,

    /// The bottom-level acceleration structure is required to be kept alive even though renderer will not
    /// directly use it. The top-level acceleration structure needs it.
    blas_map: HashMap<String, Arc<AccelerationStructure>>,
}

impl AccelerationStructures {
    /// Create new acceleration structures for the given model.
    pub fn new(
        vk: Arc<Vk>,
        mesh_instances: &[MeshInstance],
        meshes: &[Arc<Mesh>],
        batch_ray_time: f32,
    ) -> Result<Self> {
        let mut mesh_map: HashMap<String, Arc<Mesh>> = HashMap::new();
        for mesh_instance in mesh_instances.iter() {
            let mesh = meshes[mesh_instance.mesh_index].clone();
            let name = mesh.name.clone();
            mesh_map.entry(name).or_insert_with(|| mesh);
        }

        let mut vertex_buffers: HashMap<String, Subbuffer<[MeshVertex]>> = HashMap::new();
        for (name, mesh) in mesh_map.iter() {
            let buf = mesh.create_blas_vertex_buffer(vk.clone())?;
            vertex_buffers.insert(name.clone(), buf);
        }

        let mut index_buffers: HashMap<String, Subbuffer<[u32]>> = HashMap::new();
        for (name, mesh) in mesh_map.iter() {
            let buf = mesh.create_blas_index_buffer(vk.clone())?;
            index_buffers.insert(name.clone(), buf);
        }

        let mut blas_map: HashMap<String, Arc<AccelerationStructure>> = HashMap::new();
        for (name, vertex_buffer) in vertex_buffers.iter() {
            let index_buffer = index_buffers
                .get(name)
                .with_context(|| format!("Index buffer {name} not found"))?;

            let acc =
                build_acceleration_structure_triangles(vk.clone(), vertex_buffer, index_buffer)?;
            blas_map.insert(name.clone(), acc);
        }

        let as_instances = build_as_instances(mesh_instances, meshes, &blas_map, batch_ray_time)?;

        // Build the top-level acceleration structure.
        let tlas =
            unsafe { build_top_level_acceleration_structure(vk.clone(), as_instances, None) }?;

        Ok(Self { blas_map, tlas })
    }

    /// Update acceleration structures for motion blur.
    pub fn update(
        &mut self,
        vk: Arc<Vk>,
        mesh_instances: &[MeshInstance],
        meshes: &[Arc<Mesh>],
        batch_ray_time: f32,
    ) -> Result<()> {
        let as_instances =
            build_as_instances(mesh_instances, meshes, &self.blas_map, batch_ray_time)?;

        self.tlas =
            unsafe { build_top_level_acceleration_structure(vk.clone(), as_instances, None) }?;

        Ok(())
    }
}

/// A helper function to build a acceleration structure and wait for its completion.
///
/// # Safety
///
/// - If you are referencing a bottom-level acceleration structure in a top-level acceleration
///   structure, you must ensure that the bottom-level acceleration structure is kept alive.
fn build_acceleration_structure_common(
    vk: Arc<Vk>,
    geometries: AccelerationStructureGeometries,
    primitive_count: u32,
    ty: AccelerationStructureType,
    old_acceleration_structure: Option<Arc<AccelerationStructure>>,
) -> Result<Arc<AccelerationStructure>> {
    let mut as_build_geometry_info = AccelerationStructureBuildGeometryInfo {
        mode: if let Some(ref old_acc) = old_acceleration_structure {
            BuildAccelerationStructureMode::Update(old_acc.clone())
        } else {
            BuildAccelerationStructureMode::Build
        },
        flags: BuildAccelerationStructureFlags::PREFER_FAST_TRACE,
        ..AccelerationStructureBuildGeometryInfo::new(geometries)
    };

    let as_build_sizes_info = vk.device.acceleration_structure_build_sizes(
        AccelerationStructureBuildType::Device,
        &as_build_geometry_info,
        &[primitive_count],
    )?;

    // Create a memory layout so the scratch buffer address is aligned correctly for the
    // acceleration structure.
    let device_properties = vk.device.physical_device().properties();
    let min_scratch_offset = device_properties
        .min_acceleration_structure_scratch_offset_alignment
        .context(
            "Unable to get min_acceleration_structure_scratch_offset_alignment device property",
        )?
        .into();

    let scratch_buffer_size = as_build_sizes_info.build_scratch_size;

    let scratch_buffer_layout =
        DeviceLayout::from_size_alignment(scratch_buffer_size, min_scratch_offset)
            .context("Unable to create scratch buffer device layout")?;

    debug!("AS min_acceleration_structure_scratch_offset_alignment: {min_scratch_offset}");
    debug!("AS scratch buffer size: {}", scratch_buffer_size);
    debug!("AS scratch buffer layout: {:?}", scratch_buffer_layout);

    // We create a new scratch buffer for each acceleration structure for simplicity. You may want
    // to reuse scratch buffers if you need to build many acceleration structures.
    let scratch_buffer = Subbuffer::new(Buffer::new(
        vk.memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC
                | BufferUsage::SHADER_DEVICE_ADDRESS
                | BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
        scratch_buffer_layout,
    )?);

    let scratch_buffer_device_address: u64 = scratch_buffer.device_address().unwrap().into();
    debug!(
        "AS scratch buffer device addr: {scratch_buffer_device_address} is {}",
        if scratch_buffer_device_address.is_multiple_of(min_scratch_offset) {
            "aligned"
        } else {
            "NOT ALIGNED"
        }
    );

    let as_create_info = AccelerationStructureCreateInfo {
        ty,
        ..AccelerationStructureCreateInfo::new(Buffer::new_slice::<u8>(
            vk.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::ACCELERATION_STRUCTURE_STORAGE
                    | BufferUsage::SHADER_DEVICE_ADDRESS,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
            as_build_sizes_info.acceleration_structure_size,
        )?)
    };

    let acceleration = if let Some(old_acc) = old_acceleration_structure {
        old_acc.clone() // Update
    } else {
        unsafe { AccelerationStructure::new(vk.device.clone(), as_create_info) }? // Build
    };

    as_build_geometry_info.dst_acceleration_structure = Some(acceleration.clone());
    as_build_geometry_info.scratch_data = Some(scratch_buffer);

    let as_build_range_info = AccelerationStructureBuildRangeInfo {
        primitive_count,
        ..Default::default()
    };

    // For simplicity, we build a single command buffer that builds the acceleration structure,
    // then waits for its execution to complete.
    let mut builder = AutoCommandBufferBuilder::primary(
        vk.command_buffer_allocator.clone(),
        vk.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )?;

    unsafe {
        builder.build_acceleration_structure(
            as_build_geometry_info,
            iter::once(as_build_range_info).collect(),
        )?
    };

    builder
        .build()?
        .execute(vk.queue.clone())?
        .then_signal_fence_and_flush()?
        .wait(None)?;

    Ok(acceleration)
}

/// Builds a bottom level accerlation strucuture for a set of triangles.
fn build_acceleration_structure_triangles(
    vk: Arc<Vk>,
    vertex_buffer: &Subbuffer<[MeshVertex]>,
    index_buffer: &Subbuffer<[u32]>,
) -> Result<Arc<AccelerationStructure>> {
    let primitive_count = (index_buffer.len() / 3) as u32;

    // NOTE: Unfortunately the clone of vertex_buffer/index_buffer is unavoidable because of
    // AccelerationStructureGeometryTrianglesData Would be nice if we could share this data.
    let as_geometry_triangles_data = AccelerationStructureGeometryTrianglesData {
        max_vertex: vertex_buffer.len() as _,
        vertex_data: Some(vertex_buffer.clone().into_bytes()),
        index_data: Some(IndexBuffer::U32(index_buffer.clone())),
        vertex_stride: size_of::<MeshVertex>() as _,
        ..AccelerationStructureGeometryTrianglesData::new(Format::R32G32B32_SFLOAT)
    };

    let geometries = AccelerationStructureGeometries::Triangles(vec![as_geometry_triangles_data]);

    build_acceleration_structure_common(
        vk,
        geometries,
        primitive_count,
        AccelerationStructureType::BottomLevel,
        None,
    )
}

/// Builds the top level accerlation strucuture.
///
/// # Safety
///
/// - If you are referencing a bottom-level acceleration structure in a top-level acceleration
///   structure, you must ensure that the bottom-level acceleration structure is kept alive.
unsafe fn build_top_level_acceleration_structure(
    vk: Arc<Vk>,
    as_instances: Vec<AccelerationStructureInstance>,
    old_acceleration_structure: Option<Arc<AccelerationStructure>>,
) -> Result<Arc<AccelerationStructure>> {
    let primitive_count = as_instances.len() as u32;

    let instance_buffer = Buffer::from_iter(
        vk.memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::SHADER_DEVICE_ADDRESS
                | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        as_instances,
    )?;

    let as_geometry_instances_data = AccelerationStructureGeometryInstancesData::new(
        AccelerationStructureGeometryInstancesDataType::Values(Some(instance_buffer)),
    );

    let geometries = AccelerationStructureGeometries::Instances(as_geometry_instances_data);

    build_acceleration_structure_common(
        vk,
        geometries,
        primitive_count,
        AccelerationStructureType::TopLevel,
        old_acceleration_structure,
    )
}

fn build_as_instances(
    mesh_instances: &[MeshInstance],
    meshes: &[Arc<Mesh>],
    blas_map: &HashMap<String, Arc<AccelerationStructure>>,
    batch_ray_time: f32,
) -> Result<Vec<AccelerationStructureInstance>> {
    let mut as_instances: Vec<_> = Vec::new();

    for mesh_instance in mesh_instances.iter() {
        let mesh_index = mesh_instance.mesh_index;
        if mesh_index >= 16_777_216 {
            warn!("Mesh count exceeds 24 bit storage for instance_custom_index_and_mask");
        }

        // Ideally we should use this to point to materials directly. For now, just use it to
        // point to the mesh index we should be using to extract material data in the shader.
        let instance_custom_index_and_mask = Packed24_8::new(mesh_index as u32, 0xFF);

        let name = meshes[mesh_index].name.clone();
        let blas = blas_map
            .get(&name)
            .with_context(|| format!("BLAS not found {name}"))?;

        let acc = AccelerationStructureInstance {
            transform: mesh_instance.get_vulkan_acc_transform(batch_ray_time),
            acceleration_structure_reference: blas.device_address().into(),
            instance_custom_index_and_mask,
            ..Default::default()
        };
        as_instances.push(acc);
    }

    Ok(as_instances)
}
