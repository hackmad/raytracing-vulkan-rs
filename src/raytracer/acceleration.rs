use std::{iter, mem::size_of, sync::Arc};

use anyhow::Result;
use vulkano::{
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
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    sync::GpuFuture,
};

use crate::raytracer::{Mesh, Vk, shaders::closest_hit};

/// Stores the acceleration structures.
pub struct AccelerationStructures {
    /// The top-level acceleration structure.
    pub tlas: Arc<AccelerationStructure>,

    /// The bottom-level acceleration structure is required to be kept alive even though renderer will not
    /// directly use it. The top-level acceleration structure needs it.
    _blas_vec: Vec<Arc<AccelerationStructure>>,
}

impl AccelerationStructures {
    /// Create new acceleration structures for the given model.
    pub fn new(vk: Arc<Vk>, meshes: &[Mesh]) -> Result<Self> {
        let vertex_buffers = meshes
            .iter()
            .map(|mesh| mesh.create_blas_vertex_buffer(vk.clone()))
            .collect::<Result<Vec<_>>>()?;

        let index_buffers = meshes
            .iter()
            .map(|model| model.create_blas_index_buffer(vk.clone()))
            .collect::<Result<Vec<_>>>()?;

        let blas_vec = vertex_buffers
            .into_iter()
            .zip(index_buffers)
            .map(|(vertex_buffer, index_buffer)| {
                build_acceleration_structure_triangles(vk.clone(), vertex_buffer, index_buffer)
            })
            .collect::<Result<Vec<_>>>()?;

        let blas_instances = blas_vec
            .iter()
            .map(|blas| AccelerationStructureInstance {
                acceleration_structure_reference: blas.device_address().into(),
                ..Default::default()
            })
            .collect();

        // Build the top-level acceleration structure.
        let tlas = unsafe { build_top_level_acceleration_structure(vk.clone(), blas_instances) }?;

        Ok(Self {
            _blas_vec: blas_vec,
            tlas,
        })
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
) -> Result<Arc<AccelerationStructure>> {
    let mut as_build_geometry_info = AccelerationStructureBuildGeometryInfo {
        mode: BuildAccelerationStructureMode::Build,
        flags: BuildAccelerationStructureFlags::PREFER_FAST_TRACE,
        ..AccelerationStructureBuildGeometryInfo::new(geometries)
    };

    let as_build_sizes_info = vk.device.acceleration_structure_build_sizes(
        AccelerationStructureBuildType::Device,
        &as_build_geometry_info,
        &[primitive_count],
    )?;

    // We create a new scratch buffer for each acceleration structure for simplicity. You may want
    // to reuse scratch buffers if you need to build many acceleration structures.
    let scratch_buffer = Buffer::new_slice::<u8>(
        vk.memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::SHADER_DEVICE_ADDRESS | BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
        as_build_sizes_info.build_scratch_size,
    )?;

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

    let acceleration = unsafe { AccelerationStructure::new(vk.device.clone(), as_create_info) }?;

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
    vertex_buffer: Subbuffer<[closest_hit::MeshVertex]>,
    index_buffer: Subbuffer<[u32]>,
) -> Result<Arc<AccelerationStructure>> {
    let primitive_count = (index_buffer.len() / 3) as u32;

    let as_geometry_triangles_data = AccelerationStructureGeometryTrianglesData {
        max_vertex: vertex_buffer.len() as _,
        vertex_data: Some(vertex_buffer.into_bytes()),
        index_data: Some(IndexBuffer::U32(index_buffer)),
        vertex_stride: size_of::<closest_hit::MeshVertex>() as _,
        ..AccelerationStructureGeometryTrianglesData::new(Format::R32G32B32_SFLOAT)
    };

    let geometries = AccelerationStructureGeometries::Triangles(vec![as_geometry_triangles_data]);

    build_acceleration_structure_common(
        vk,
        geometries,
        primitive_count,
        AccelerationStructureType::BottomLevel,
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
    )
}
