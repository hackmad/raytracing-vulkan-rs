use glam::{Mat4, Vec3};
use std::{f32::consts::FRAC_PI_2, iter, mem::size_of, sync::Arc};
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
        allocator::{
            CommandBufferAllocator, StandardCommandBufferAllocator,
            StandardCommandBufferAllocatorCreateInfo,
        },
    },
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet,
        allocator::StandardDescriptorSetAllocator,
        layout::{
            DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
            DescriptorType,
        },
    },
    device::{Device, Queue},
    format::Format,
    image::view::ImageView,
    memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter},
    pipeline::{
        PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
        layout::PipelineLayoutCreateInfo,
        ray_tracing::{
            RayTracingPipeline, RayTracingPipelineCreateInfo, RayTracingShaderGroupCreateInfo,
            ShaderBindingTable,
        },
    },
    shader::ShaderStages,
    sync::GpuFuture,
};

use super::{
    geometry::VertexData,
    model::Model,
    shaders::{ShaderModules, closest_hit, ray_gen},
};

pub struct Scene {
    queue: Arc<Queue>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    tlas_descriptor_set: Arc<DescriptorSet>,
    mesh_data_descriptor_set: Arc<DescriptorSet>,
    pipeline_layout: Arc<PipelineLayout>,
    shader_binding_table: ShaderBindingTable,
    pipeline: Arc<RayTracingPipeline>,
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,

    // The bottom-level acceleration structure is required to be kept alive
    // as we reference it in the top-level acceleration structure.
    _blas: Arc<AccelerationStructure>,
    _tlas: Arc<AccelerationStructure>,
}

impl Scene {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        memory_allocator: Arc<dyn MemoryAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        model: &Model,
    ) -> Self {
        let pipeline_layout = create_pipeline_layout(device.clone());
        let pipeline = create_raytracing_pipeline(device.clone(), pipeline_layout.clone());

        let vertex_buffer = model
            .create_vertex_buffer(memory_allocator.clone())
            .unwrap();
        let vertex_buffer_device_address = vertex_buffer.device_address().unwrap();

        let index_buffer = model.create_index_buffer(memory_allocator.clone()).unwrap();
        let index_buffer_device_address = index_buffer.device_address().unwrap();

        // Create an allocator for command-buffer data
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            queue.device().clone(),
            StandardCommandBufferAllocatorCreateInfo {
                secondary_buffer_count: 32,
                ..Default::default()
            },
        ));

        // Build the bottom-level acceleration structure and then the top-level acceleration
        // structure. Acceleration structures are used to accelerate ray tracing. The bottom-level
        // acceleration structure contains the geometry data. The top-level acceleration structure
        // contains the instances of the bottom-level acceleration structures. In our shader, we
        // will trace rays against the top-level acceleration structure.
        let blas = build_acceleration_structure_triangles(
            vertex_buffer,
            index_buffer,
            memory_allocator.clone(),
            command_buffer_allocator.clone(),
            device.clone(),
            queue.clone(),
        );

        let tlas = unsafe {
            build_top_level_acceleration_structure(
                vec![AccelerationStructureInstance {
                    acceleration_structure_reference: blas.device_address().into(),
                    ..Default::default()
                }],
                memory_allocator.clone(),
                command_buffer_allocator.clone(),
                device.clone(),
                queue.clone(),
            )
        };

        // For now the acceleration structure is non-changing. We can create its descriptor set
        // and clone it later during render.
        let tlas_descriptor_set = DescriptorSet::new(
            descriptor_set_allocator.clone(),
            pipeline_layout.set_layouts()[0].clone(),
            [WriteDescriptorSet::acceleration_structure(0, tlas.clone())],
            [],
        )
        .unwrap();

        // Mesh data references for vertex and index buffer.
        let mesh_data = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::SHADER_DEVICE_ADDRESS
                    | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            [closest_hit::MeshData {
                vertexBufferAddress: vertex_buffer_device_address.into(),
                indexBufferAddress: index_buffer_device_address.into(),
            }],
        )
        .unwrap();

        // Mesh data won't change either. We can create its descriptor set and clone it later
        // during render.
        let mesh_data_descriptor_set = DescriptorSet::new(
            descriptor_set_allocator.clone(),
            pipeline_layout.set_layouts()[3].clone(),
            [WriteDescriptorSet::buffer(0, mesh_data.clone())],
            [],
        )
        .unwrap();

        // Create the shader binding table.
        let shader_binding_table =
            ShaderBindingTable::new(memory_allocator.clone(), &pipeline).unwrap();

        Scene {
            queue,
            descriptor_set_allocator,
            tlas_descriptor_set,
            mesh_data_descriptor_set,
            pipeline_layout,
            shader_binding_table,
            pipeline,
            memory_allocator,
            command_buffer_allocator,
            _blas: blas,
            _tlas: tlas,
        }
    }

    pub fn render(
        &self,
        before_future: Box<dyn GpuFuture>,
        image_view: Arc<ImageView>,
    ) -> Box<dyn GpuFuture> {
        let dimensions = image_view.image().extent();

        let aspect = dimensions[0] as f32 / dimensions[1] as f32;

        let proj = Mat4::perspective_rh(FRAC_PI_2, aspect, 0.01, 100.0);
        let view = Mat4::look_at_rh(
            Vec3::new(5.5, 3.5, -4.5),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
        );

        let uniform_buffer = Buffer::from_data(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            ray_gen::Camera {
                viewProj: (proj * view).to_cols_array_2d(),
                viewInverse: view.inverse().to_cols_array_2d(),
                projInverse: proj.inverse().to_cols_array_2d(),
            },
        )
        .unwrap();

        let uniform_buffer_descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            self.pipeline_layout.set_layouts()[1].clone(),
            [WriteDescriptorSet::buffer(0, uniform_buffer.clone())],
            [],
        )
        .unwrap();

        let storage_image_descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            self.pipeline_layout.set_layouts()[2].clone(),
            [WriteDescriptorSet::image_view(0, image_view.clone())],
            [],
        )
        .unwrap();

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .bind_descriptor_sets(
                PipelineBindPoint::RayTracing,
                self.pipeline_layout.clone(),
                0,
                vec![
                    self.tlas_descriptor_set.clone(),
                    uniform_buffer_descriptor_set,
                    storage_image_descriptor_set,
                    self.mesh_data_descriptor_set.clone(),
                ],
            )
            .unwrap()
            .bind_pipeline_ray_tracing(self.pipeline.clone())
            .unwrap();

        unsafe {
            builder
                .trace_rays(self.shader_binding_table.addresses().clone(), dimensions)
                .unwrap();
        }

        let command_buffer = builder.build().unwrap();

        let after_future = before_future
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap();

        after_future.boxed()
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
) -> Arc<AccelerationStructure> {
    let mut as_build_geometry_info = AccelerationStructureBuildGeometryInfo {
        mode: BuildAccelerationStructureMode::Build,
        flags: BuildAccelerationStructureFlags::PREFER_FAST_TRACE,
        ..AccelerationStructureBuildGeometryInfo::new(geometries)
    };

    let as_build_sizes_info = device
        .acceleration_structure_build_sizes(
            AccelerationStructureBuildType::Device,
            &as_build_geometry_info,
            &[primitive_count],
        )
        .unwrap();

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
    )
    .unwrap();

    let as_create_info = AccelerationStructureCreateInfo {
        ty,
        ..AccelerationStructureCreateInfo::new(
            Buffer::new_slice::<u8>(
                memory_allocator,
                BufferCreateInfo {
                    usage: BufferUsage::ACCELERATION_STRUCTURE_STORAGE
                        | BufferUsage::SHADER_DEVICE_ADDRESS,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
                as_build_sizes_info.acceleration_structure_size,
            )
            .unwrap(),
        )
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
        .build()
        .unwrap()
        .execute(queue)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    acceleration
}

fn build_acceleration_structure_triangles(
    vertex_buffer: Subbuffer<[VertexData]>,
    index_buffer: Subbuffer<[u32]>,
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: Arc<dyn CommandBufferAllocator>,
    device: Arc<Device>,
    queue: Arc<Queue>,
) -> Arc<AccelerationStructure> {
    let primitive_count = (vertex_buffer.len() / 3) as u32;

    let as_geometry_triangles_data = AccelerationStructureGeometryTrianglesData {
        max_vertex: vertex_buffer.len() as _,
        vertex_data: Some(vertex_buffer.into_bytes()),
        index_data: Some(IndexBuffer::U32(index_buffer)),
        vertex_stride: size_of::<VertexData>() as _,
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
) -> Arc<AccelerationStructure> {
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
    )
    .unwrap();

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

/// Create a raytracing pipeline.
fn create_raytracing_pipeline(
    device: Arc<Device>,
    pipeline_layout: Arc<PipelineLayout>,
) -> Arc<RayTracingPipeline> {
    // Load the shader modules.
    let shader_modules = ShaderModules::load(device.clone());

    // Make a list of the shader stages that the pipeline will have.
    let stages = [
        PipelineShaderStageCreateInfo::new(shader_modules.ray_gen),
        PipelineShaderStageCreateInfo::new(shader_modules.ray_miss),
        PipelineShaderStageCreateInfo::new(shader_modules.closest_hit),
    ];

    // Define the shader groups that will eventually turn into the shader binding table.
    // The numbers are the indices of the stages in the `stages` array.
    let groups = [
        RayTracingShaderGroupCreateInfo::General { general_shader: 0 },
        RayTracingShaderGroupCreateInfo::General { general_shader: 1 },
        RayTracingShaderGroupCreateInfo::TrianglesHit {
            closest_hit_shader: Some(2),
            any_hit_shader: None,
        },
    ];

    RayTracingPipeline::new(
        device.clone(),
        None,
        RayTracingPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            groups: groups.into_iter().collect(),
            max_pipeline_ray_recursion_depth: 1,
            ..RayTracingPipelineCreateInfo::layout(pipeline_layout.clone())
        },
    )
    .unwrap()
}

/// Create the pipeline layout. This will contain the descriptor sets matching the layouts in
/// ray_gen.glsl shader.
fn create_pipeline_layout(device: Arc<Device>) -> Arc<PipelineLayout> {
    PipelineLayout::new(
        device.clone(),
        PipelineLayoutCreateInfo {
            set_layouts: vec![
                // Top level acceleration structure.
                DescriptorSetLayout::new(
                    device.clone(),
                    DescriptorSetLayoutCreateInfo {
                        bindings: [(
                            0,
                            DescriptorSetLayoutBinding {
                                stages: ShaderStages::RAYGEN,
                                ..DescriptorSetLayoutBinding::descriptor_type(
                                    DescriptorType::AccelerationStructure,
                                )
                            },
                        )]
                        .into_iter()
                        .collect(),
                        ..Default::default()
                    },
                )
                .unwrap(),
                // Uniform buffer containing camera matrices.
                DescriptorSetLayout::new(
                    device.clone(),
                    DescriptorSetLayoutCreateInfo {
                        bindings: [(
                            0,
                            DescriptorSetLayoutBinding {
                                stages: ShaderStages::RAYGEN,
                                ..DescriptorSetLayoutBinding::descriptor_type(
                                    DescriptorType::UniformBuffer,
                                )
                            },
                        )]
                        .into_iter()
                        .collect(),
                        ..Default::default()
                    },
                )
                .unwrap(),
                // Storage image for the render.
                DescriptorSetLayout::new(
                    device.clone(),
                    DescriptorSetLayoutCreateInfo {
                        bindings: [(
                            0,
                            DescriptorSetLayoutBinding {
                                stages: ShaderStages::RAYGEN,
                                ..DescriptorSetLayoutBinding::descriptor_type(
                                    DescriptorType::StorageImage,
                                )
                            },
                        )]
                        .into_iter()
                        .collect(),
                        ..Default::default()
                    },
                )
                .unwrap(),
                // Storage buffer for the mesh data references.
                DescriptorSetLayout::new(
                    device.clone(),
                    DescriptorSetLayoutCreateInfo {
                        bindings: [(
                            0,
                            DescriptorSetLayoutBinding {
                                stages: ShaderStages::CLOSEST_HIT,
                                ..DescriptorSetLayoutBinding::descriptor_type(
                                    DescriptorType::StorageBuffer,
                                )
                            },
                        )]
                        .into_iter()
                        .collect(),
                        ..Default::default()
                    },
                )
                .unwrap(),
            ],
            ..Default::default()
        },
    )
    .unwrap()
}
