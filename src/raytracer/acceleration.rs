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
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBufferAbstract,
        allocator::CommandBufferAllocator,
    },
    device::{Device, Queue},
    format::Format,
    memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter},
    sync::GpuFuture,
};

use super::{Model, shaders::closest_hit};

/// Stores the acceleration structures.
pub struct AccelerationStructures {
    /// The top-level acceleration structure.
    pub tlas: Arc<AccelerationStructure>,

    /// The bottom-level acceleration structure is required to be kept alive even though renderer will not
    /// directly use it. The top-level acceleration structure needs it.
    _blas_vec: Vec<Arc<AccelerationStructure>>,
}

impl AccelerationStructures {
    pub fn new(
        models: &[Model],
        memory_allocator: Arc<dyn MemoryAllocator>,
        command_buffer_allocator: Arc<dyn CommandBufferAllocator>,
        device: Arc<Device>,
        queue: Arc<Queue>,
    ) -> Result<Self> {
        let vertex_buffers: Vec<_> = models
            .iter()
            .map(|model| {
                model
                    .create_blas_vertex_buffer(
                        memory_allocator.clone(),
                        command_buffer_allocator.clone(),
                        queue.clone(),
                    )
                    .unwrap()
            })
            .collect();

        let index_buffers: Vec<_> = models
            .iter()
            .map(|model| {
                model
                    .create_blas_index_buffer(
                        memory_allocator.clone(),
                        command_buffer_allocator.clone(),
                        queue.clone(),
                    )
                    .unwrap()
            })
            .collect();

        let blas_vec_result: Result<Vec<_>> = vertex_buffers
            .into_iter()
            .zip(index_buffers)
            .map(|(vertex_buffer, index_buffer)| {
                build_acceleration_structure_triangles(
                    vertex_buffer,
                    index_buffer,
                    memory_allocator.clone(),
                    command_buffer_allocator.clone(),
                    device.clone(),
                    queue.clone(),
                )
            })
            .collect();
        let blas_vec = blas_vec_result?;

        let blas_instances = blas_vec
            .iter()
            .map(|blas| AccelerationStructureInstance {
                acceleration_structure_reference: blas.device_address().into(),
                ..Default::default()
            })
            .collect();

        // Build the top-level acceleration structure.
        let tlas = unsafe {
            build_top_level_acceleration_structure(
                blas_instances,
                memory_allocator.clone(),
                command_buffer_allocator.clone(),
                device.clone(),
                queue.clone(),
            )
        }?;

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
    geometries: AccelerationStructureGeometries,
    primitive_count: u32,
    ty: AccelerationStructureType,
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: Arc<dyn CommandBufferAllocator>,
    device: Arc<Device>,
    queue: Arc<Queue>,
) -> Result<Arc<AccelerationStructure>> {
    let mut as_build_geometry_info = AccelerationStructureBuildGeometryInfo {
        mode: BuildAccelerationStructureMode::Build,
        flags: BuildAccelerationStructureFlags::PREFER_FAST_TRACE,
        ..AccelerationStructureBuildGeometryInfo::new(geometries)
    };

    let as_build_sizes_info = device.acceleration_structure_build_sizes(
        AccelerationStructureBuildType::Device,
        &as_build_geometry_info,
        &[primitive_count],
    )?;

    // We create a new scratch buffer for each acceleration structure for simplicity. You may want
    // to reuse scratch buffers if you need to build many acceleration structures.
    let scratch_buffer = Buffer::new_slice::<u8>(
        memory_allocator.clone(),
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
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::ACCELERATION_STRUCTURE_STORAGE
                    | BufferUsage::SHADER_DEVICE_ADDRESS,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
            as_build_sizes_info.acceleration_structure_size,
        )?)
    };

    let acceleration = unsafe { AccelerationStructure::new(device, as_create_info) }.unwrap();

    as_build_geometry_info.dst_acceleration_structure = Some(acceleration.clone());
    as_build_geometry_info.scratch_data = Some(scratch_buffer);

    let as_build_range_info = AccelerationStructureBuildRangeInfo {
        primitive_count,
        ..Default::default()
    };

    // For simplicity, we build a single command buffer that builds the acceleration structure,
    // then waits for its execution to complete.
    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    unsafe {
        builder
            .build_acceleration_structure(
                as_build_geometry_info,
                iter::once(as_build_range_info).collect(),
            )
            .unwrap()
    };

    builder
        .build()?
        .execute(queue)?
        .then_signal_fence_and_flush()?
        .wait(None)?;

    Ok(acceleration)
}

fn build_acceleration_structure_triangles(
    vertex_buffer: Subbuffer<[closest_hit::MeshVertex]>,
    index_buffer: Subbuffer<[u32]>,
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: Arc<dyn CommandBufferAllocator>,
    device: Arc<Device>,
    queue: Arc<Queue>,
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
        geometries,
        primitive_count,
        AccelerationStructureType::BottomLevel,
        memory_allocator,
        command_buffer_allocator,
        device,
        queue,
    )
}

unsafe fn build_top_level_acceleration_structure(
    as_instances: Vec<AccelerationStructureInstance>,
    allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: Arc<dyn CommandBufferAllocator>,
    device: Arc<Device>,
    queue: Arc<Queue>,
) -> Result<Arc<AccelerationStructure>> {
    let primitive_count = as_instances.len() as u32;

    let instance_buffer = Buffer::from_iter(
        allocator.clone(),
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
        geometries,
        primitive_count,
        AccelerationStructureType::TopLevel,
        allocator,
        command_buffer_allocator,
        device,
        queue,
    )
}
