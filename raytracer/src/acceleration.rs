use std::sync::Arc;

use crate::MeshGeometryBuffers;
use anyhow::Result;
use ash::{
    khr,
    vk::{self, Packed24_8},
};
use vulkan::{Buffer, CommandBuffer, NO_FENCE, VulkanContext};

#[rustfmt::skip]
pub const IDENTITY_TRANSFORM: [f32; 12] = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
];

/// Stores the acceleration structures.
pub struct AccelerationStructures {
    _blas_vec: Vec<AccelerationStructure>,
    _blas_instances: Vec<vk::AccelerationStructureInstanceKHR>,
    pub tlas: AccelerationStructure,
}

impl AccelerationStructures {
    /// Create new acceleration structures for the given model.
    pub fn new(
        context: Arc<VulkanContext>,
        mesh_geometry_buffers: &[MeshGeometryBuffers],
    ) -> Result<Self> {
        let as_loader = Arc::new(khr::acceleration_structure::Device::new(
            &context.instance,
            &context.device,
        ));

        let blas_vec = mesh_geometry_buffers
            .iter()
            .map(|geometry_buffers| {
                AccelerationStructure::new_bottom_level_accleration_structure(
                    context.clone(),
                    as_loader.clone(),
                    geometry_buffers,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let blas_instances = blas_vec
            .iter()
            .enumerate()
            .map(|(index, blas)| blas.create_instance(index as _, IDENTITY_TRANSFORM))
            .collect::<Vec<_>>();

        let blas_instance_count = blas_instances.len();

        let blas_instance_buffer_size =
            std::mem::size_of::<vk::AccelerationStructureInstanceKHR>() * blas_instance_count;

        let mut blas_instance_buffer = Buffer::new(
            context.clone(),
            blas_instance_buffer_size as _,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        blas_instance_buffer.store(&blas_instances)?;

        let tlas = AccelerationStructure::new_top_level_accleration_structure(
            context.clone(),
            as_loader,
            &blas_instance_buffer,
            blas_instance_count,
        )?;

        Ok(Self {
            _blas_vec: blas_vec,
            _blas_instances: blas_instances,
            tlas,
        })
    }
}

pub struct AccelerationStructure {
    as_loader: Arc<khr::acceleration_structure::Device>,
    pub acceleration_structure: vk::AccelerationStructureKHR,
    handle: u64,
    _buffer: Buffer,
}

impl AccelerationStructure {
    fn new(
        context: Arc<VulkanContext>,
        as_loader: Arc<khr::acceleration_structure::Device>,
        ty: vk::AccelerationStructureTypeKHR,
        geometries: &[vk::AccelerationStructureGeometryKHR],
        instance_count: usize,
    ) -> Result<Self> {
        let build_range_info = vk::AccelerationStructureBuildRangeInfoKHR::default()
            .first_vertex(0)
            .primitive_count(instance_count as u32)
            .primitive_offset(0)
            .transform_offset(0);

        let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(geometries)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .ty(ty);

        let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            as_loader.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &[build_range_info.primitive_count],
                &mut size_info,
            )
        };

        let buffer = Buffer::new(
            context.clone(),
            size_info.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let as_create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .ty(build_info.ty)
            .size(size_info.acceleration_structure_size)
            .buffer(buffer.buffer)
            .offset(0);

        let acceleration_structure =
            unsafe { as_loader.create_acceleration_structure(&as_create_info, None)? };

        build_info.dst_acceleration_structure = acceleration_structure;

        let scratch_buffer = Buffer::new(
            context.clone(),
            size_info.build_scratch_size,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        build_info.scratch_data = vk::DeviceOrHostAddressKHR {
            device_address: scratch_buffer.get_buffer_device_address(),
        };

        let command_buffer = CommandBuffer::new(context.clone())?;
        command_buffer.begin_one_time_submit()?;

        let memory_barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR);

        command_buffer.memory_barrier(
            memory_barrier,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::DependencyFlags::empty(),
        );

        unsafe {
            as_loader.cmd_build_acceleration_structures(
                command_buffer.get(),
                &[build_info],
                &[&[build_range_info]],
            );
        }

        command_buffer.end()?;

        command_buffer.submit(None, &NO_FENCE)?;

        let handle = {
            let as_addr_info = vk::AccelerationStructureDeviceAddressInfoKHR::default()
                .acceleration_structure(acceleration_structure);
            unsafe { as_loader.get_acceleration_structure_device_address(&as_addr_info) }
        };

        Ok(Self {
            as_loader,
            acceleration_structure,
            handle,
            _buffer: buffer,
        })
    }

    fn new_bottom_level_accleration_structure(
        context: Arc<VulkanContext>,
        as_loader: Arc<khr::acceleration_structure::Device>,
        mesh_geometry_buffers: &MeshGeometryBuffers,
    ) -> Result<AccelerationStructure> {
        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                    // Vertices
                    .vertex_data(vk::DeviceOrHostAddressConstKHR {
                        device_address: mesh_geometry_buffers
                            .vertex_buffer
                            .get_buffer_device_address(),
                    })
                    .max_vertex(mesh_geometry_buffers.vertex_count as u32 - 1)
                    .vertex_stride(mesh_geometry_buffers.vertex_stride as u64)
                    .vertex_format(vk::Format::R32G32B32_SFLOAT)
                    //
                    // Indices
                    .index_data(vk::DeviceOrHostAddressConstKHR {
                        device_address: mesh_geometry_buffers
                            .index_buffer
                            .get_buffer_device_address(),
                    })
                    .index_type(vk::IndexType::UINT32),
            })
            .flags(vk::GeometryFlagsKHR::OPAQUE);

        Self::new(
            context.clone(),
            as_loader,
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
            &[geometry],
            mesh_geometry_buffers.index_count / 3,
        )
    }

    fn new_top_level_accleration_structure(
        context: Arc<VulkanContext>,
        as_loader: Arc<khr::acceleration_structure::Device>,
        blas_instance_buffer: &Buffer,
        blas_instance_count: usize,
    ) -> Result<AccelerationStructure> {
        let tlas_instances = vk::AccelerationStructureGeometryInstancesDataKHR::default()
            .array_of_pointers(false)
            .data(vk::DeviceOrHostAddressConstKHR {
                device_address: blas_instance_buffer.get_buffer_device_address(),
            });

        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: tlas_instances,
            });

        Self::new(
            context.clone(),
            as_loader,
            vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            &[geometry],
            blas_instance_count,
        )
    }

    // Use this to create transformed instances for the same mesh. This should be used when
    // generating the bottom level acceleration structure.
    fn create_instance(
        &self,
        index: u32,
        transform: [f32; 12],
    ) -> vk::AccelerationStructureInstanceKHR {
        vk::AccelerationStructureInstanceKHR {
            transform: vk::TransformMatrixKHR { matrix: transform },
            instance_custom_index_and_mask: Packed24_8::new(index, 0xff),
            instance_shader_binding_table_record_offset_and_flags: Packed24_8::new(
                0, // RAY_GEN
                vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
            ),
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                device_handle: self.handle,
            },
        }
    }
}

impl Drop for AccelerationStructure {
    fn drop(&mut self) {
        unsafe {
            self.as_loader
                .destroy_acceleration_structure(self.acceleration_structure, None);
        }
    }
}
