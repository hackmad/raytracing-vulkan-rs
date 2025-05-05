use anyhow::Result;
use image::{GenericImageView, ImageReader};
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};
use vulkano::{
    DeviceSize,
    buffer::{Buffer, BufferCreateInfo, BufferUsage, IndexBuffer, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo,
        PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract, allocator::CommandBufferAllocator,
    },
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet,
        allocator::DescriptorSetAllocator,
        layout::{
            DescriptorBindingFlags, DescriptorSetLayout, DescriptorSetLayoutBinding,
            DescriptorSetLayoutCreateInfo, DescriptorType,
        },
    },
    device::{Device, Queue},
    format::Format,
    image::{
        Image, ImageCreateInfo, ImageType, ImageUsage,
        sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
        view::ImageView,
    },
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
    Camera, MaterialPropertyData, MaterialPropertyDataEnum,
    acceleration::AccelerationStructures,
    model::Model,
    shaders::{ShaderModules, closest_hit, ray_gen},
};

pub struct Scene {
    queue: Arc<Queue>,
    descriptor_set_allocator: Arc<dyn DescriptorSetAllocator>,
    tlas_descriptor_set: Arc<DescriptorSet>,
    mesh_data_descriptor_set: Arc<DescriptorSet>,
    textures_descriptor_set: Arc<DescriptorSet>,
    pipeline_layout: Arc<PipelineLayout>,
    shader_binding_table: ShaderBindingTable,
    pipeline: Arc<RayTracingPipeline>,
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: Arc<dyn CommandBufferAllocator>,

    acceleration_structures: AccelerationStructures,

    camera: Arc<RwLock<dyn Camera>>,
}

impl Scene {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        memory_allocator: Arc<dyn MemoryAllocator>,
        descriptor_set_allocator: Arc<dyn DescriptorSetAllocator>,
        command_buffer_allocator: Arc<dyn CommandBufferAllocator>,
        models: &[Model],
        camera: Arc<RwLock<dyn Camera>>,
    ) -> Self {
        // Load shader modules
        let (stages, groups) = load_shader_modules(device.clone());

        // Load Textures.
        let (textures, texture_indices) = load_textures(
            models,
            memory_allocator.clone(),
            command_buffer_allocator.clone(),
            queue.clone(),
        )
        .unwrap();

        let pipeline_layout = create_pipeline_layout(device.clone(), textures.len() as u32);

        let pipeline =
            create_raytracing_pipeline(device.clone(), pipeline_layout.clone(), &stages, &groups);

        let layouts = pipeline_layout.set_layouts();

        // For now the acceleration structure is non-changing. We can create its descriptor set
        // and clone it later during render.
        let acceleration_structures = AccelerationStructures::new(
            models,
            memory_allocator.clone(),
            command_buffer_allocator.clone(),
            device.clone(),
            queue.clone(),
        )
        .unwrap();

        let tlas_descriptor_set = DescriptorSet::new(
            descriptor_set_allocator.clone(),
            layouts[TLAS_LAYOUT].clone(),
            [WriteDescriptorSet::acceleration_structure(
                0,
                acceleration_structures.tlas.clone(),
            )],
            [],
        )
        .unwrap();

        // Create storage for mesh data references.
        let mesh_vertices_storage_buffers: Vec<_> = models
            .iter()
            .map(|model| {
                model
                    .create_vertices_storage_buffer(
                        memory_allocator.clone(),
                        command_buffer_allocator.clone(),
                        queue.clone(),
                    )
                    .unwrap()
            })
            .collect();

        let mesh_indices_storage_buffers: Vec<_> = models
            .iter()
            .map(|model| {
                model
                    .create_indices_storage_buffer(
                        memory_allocator.clone(),
                        command_buffer_allocator.clone(),
                        queue.clone(),
                    )
                    .unwrap()
            })
            .collect();

        let mesh_vertices_buffer_device_addresses: Vec<u64> = mesh_vertices_storage_buffers
            .iter()
            .map(|buf| buf.device_address().unwrap().into())
            .collect();

        let mesh_indices_buffer_device_addresses: Vec<u64> = mesh_indices_storage_buffers
            .iter()
            .map(|buf| buf.device_address().unwrap().into())
            .collect();

        let meshes = mesh_vertices_buffer_device_addresses
            .into_iter()
            .zip(mesh_indices_buffer_device_addresses)
            .map(|(vertices_ref, indices_ref)| closest_hit::Mesh {
                vertices_ref,
                indices_ref,
            });

        let mesh_data = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            meshes,
        )
        .unwrap();

        // Mesh data won't change either. We can create its descriptor set and clone it later
        // during render.
        let mesh_data_descriptor_set = DescriptorSet::new(
            descriptor_set_allocator.clone(),
            layouts[MESH_DATA_LAYOUT].clone(),
            [WriteDescriptorSet::buffer(0, mesh_data)],
            [],
        )
        .unwrap();

        // Textures + Sampler
        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                //mipmap_mode: SamplerMipmapMode::Nearest,
                //mip_lod_bias: 0.0,
                ..Default::default()
            },
        )
        .unwrap();

        let textures_descriptor_set = DescriptorSet::new_variable(
            descriptor_set_allocator.clone(),
            layouts[SAMPLERS_AND_TEXTURES_LAYOUT].clone(),
            textures.len() as u32,
            [
                WriteDescriptorSet::sampler(0, sampler.clone()),
                WriteDescriptorSet::image_view_array(1, 0, textures),
            ],
            [],
        )
        .unwrap();

        // Materials
        /*
        for model in models.iter() {
            if let Some(material) = &model.material {
                let _diffuse = get_material_data(&material.diffuse, &texture_indices);
            }
        }
        */

        // Create the shader binding table.
        let shader_binding_table =
            ShaderBindingTable::new(memory_allocator.clone(), &pipeline).unwrap();

        Scene {
            queue,
            descriptor_set_allocator,
            tlas_descriptor_set,
            mesh_data_descriptor_set,
            textures_descriptor_set,
            pipeline_layout,
            shader_binding_table,
            pipeline,
            memory_allocator,
            command_buffer_allocator,
            acceleration_structures,
            camera,
        }
    }

    pub fn update_window_size(&mut self, window_size: [f32; 2]) {
        let mut camera = self.camera.write().unwrap();
        camera.update_image_size(window_size[0] as u32, window_size[1] as u32);
    }

    pub fn render(
        &self,
        before_future: Box<dyn GpuFuture>,
        image_view: Arc<ImageView>,
    ) -> Box<dyn GpuFuture> {
        let dimensions = image_view.image().extent();

        let camera = self.camera.read().unwrap();

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
                viewProj: (camera.get_projection_matrix() * camera.get_view_matrix())
                    .to_cols_array_2d(),
                viewInverse: camera.get_view_inverse_matrix().to_cols_array_2d(),
                projInverse: camera.get_projection_inverse_matrix().to_cols_array_2d(),
            },
        )
        .unwrap();

        let layouts = self.pipeline_layout.set_layouts();

        let uniform_buffer_descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            layouts[UNIFORM_BUFFER_LAYOUT].clone(),
            [WriteDescriptorSet::buffer(0, uniform_buffer)],
            [],
        )
        .unwrap();

        let render_image_descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            layouts[RENDER_IMAGE_LAYOUT].clone(),
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
                    render_image_descriptor_set,
                    self.mesh_data_descriptor_set.clone(),
                    self.textures_descriptor_set.clone(),
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

/// Load shader modules
fn load_shader_modules(
    device: Arc<Device>,
) -> (
    Vec<PipelineShaderStageCreateInfo>,
    Vec<RayTracingShaderGroupCreateInfo>,
) {
    // Load the shader modules.
    let shader_modules = ShaderModules::load(device.clone());

    // Make a list of the shader stages that the pipeline will have.
    let stages = vec![
        PipelineShaderStageCreateInfo::new(shader_modules.ray_gen),
        PipelineShaderStageCreateInfo::new(shader_modules.ray_miss),
        PipelineShaderStageCreateInfo::new(shader_modules.closest_hit),
    ];

    // Define the shader groups that will eventually turn into the shader binding table.
    // The numbers are the indices of the stages in the `stages` array.
    let groups = vec![
        RayTracingShaderGroupCreateInfo::General { general_shader: 0 },
        RayTracingShaderGroupCreateInfo::General { general_shader: 1 },
        RayTracingShaderGroupCreateInfo::TrianglesHit {
            closest_hit_shader: Some(2),
            any_hit_shader: None,
        },
    ];

    (stages, groups)
}

/// Create a raytracing pipeline.
fn create_raytracing_pipeline(
    device: Arc<Device>,
    pipeline_layout: Arc<PipelineLayout>,
    stages: &[PipelineShaderStageCreateInfo],
    groups: &[RayTracingShaderGroupCreateInfo],
) -> Arc<RayTracingPipeline> {
    RayTracingPipeline::new(
        device.clone(),
        None,
        RayTracingPipelineCreateInfo {
            stages: stages.into(),
            groups: groups.into(),
            max_pipeline_ray_recursion_depth: 1,
            ..RayTracingPipelineCreateInfo::layout(pipeline_layout.clone())
        },
    )
    .unwrap()
}

/// Pipeline layout for top level acceleration structure.
fn create_tlas_layout(device: Arc<Device>) -> Arc<DescriptorSetLayout> {
    DescriptorSetLayout::new(
        device,
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
    .unwrap()
}

/// Pipeline layout for uniform buffer containing camera matrices.
fn create_uniform_buffer_layout(device: Arc<Device>) -> Arc<DescriptorSetLayout> {
    DescriptorSetLayout::new(
        device,
        DescriptorSetLayoutCreateInfo {
            bindings: [(
                0,
                DescriptorSetLayoutBinding {
                    stages: ShaderStages::RAYGEN,
                    ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBuffer)
                },
            )]
            .into_iter()
            .collect(),
            ..Default::default()
        },
    )
    .unwrap()
}

/// Pipeline layout for the render image storage buffer.
fn create_render_image_layout(device: Arc<Device>) -> Arc<DescriptorSetLayout> {
    DescriptorSetLayout::new(
        device.clone(),
        DescriptorSetLayoutCreateInfo {
            bindings: [(
                0,
                DescriptorSetLayoutBinding {
                    stages: ShaderStages::RAYGEN,
                    ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageImage)
                },
            )]
            .into_iter()
            .collect(),
            ..Default::default()
        },
    )
    .unwrap()
}

/// Piepline layout for mesh data references storage buffer.
fn create_mesh_data_layout(device: Arc<Device>) -> Arc<DescriptorSetLayout> {
    DescriptorSetLayout::new(
        device.clone(),
        DescriptorSetLayoutCreateInfo {
            bindings: [(
                0,
                DescriptorSetLayoutBinding {
                    stages: ShaderStages::CLOSEST_HIT,
                    ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
                },
            )]
            .into_iter()
            .collect(),
            ..Default::default()
        },
    )
    .unwrap()
}

/// Pipeline layout for sampler and textures.
fn create_sample_and_textures_layout(
    device: Arc<Device>,
    texture_count: u32,
) -> Arc<DescriptorSetLayout> {
    DescriptorSetLayout::new(
        device.clone(),
        DescriptorSetLayoutCreateInfo {
            bindings: [
                (
                    0,
                    DescriptorSetLayoutBinding {
                        stages: ShaderStages::CLOSEST_HIT,
                        ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::Sampler)
                    },
                ),
                (
                    1,
                    DescriptorSetLayoutBinding {
                        stages: ShaderStages::CLOSEST_HIT,
                        binding_flags: DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT,
                        descriptor_count: texture_count,
                        ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::SampledImage)
                    },
                ),
            ]
            .into_iter()
            .collect(),
            ..Default::default()
        },
    )
    .unwrap()
}

// These will help referencing the layout indices. Keep in sync with create_pipeline_layout().
const TLAS_LAYOUT: usize = 0;
const UNIFORM_BUFFER_LAYOUT: usize = 1;
const RENDER_IMAGE_LAYOUT: usize = 2;
const MESH_DATA_LAYOUT: usize = 3;
const SAMPLERS_AND_TEXTURES_LAYOUT: usize = 4;

/// Create the pipeline layout. This will contain the descriptor sets matching the layouts in
/// ray_gen.glsl shader.
fn create_pipeline_layout(device: Arc<Device>, texture_count: u32) -> Arc<PipelineLayout> {
    PipelineLayout::new(
        device.clone(),
        PipelineLayoutCreateInfo {
            set_layouts: vec![
                create_tlas_layout(device.clone()),
                create_uniform_buffer_layout(device.clone()),
                create_render_image_layout(device.clone()),
                create_mesh_data_layout(device.clone()),
                create_sample_and_textures_layout(device.clone(), texture_count),
            ],
            ..Default::default()
        },
    )
    .unwrap()
}

/// Loads the image texture into an new image view.
/// Assumes image has alpha.
fn load_texture(
    path: &str,
    memory_allocator: Arc<dyn MemoryAllocator>,
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
) -> Result<Arc<ImageView>> {
    println!("Loading texture {path}...");

    let img = ImageReader::open(path)?.with_guessed_format()?.decode()?;
    let (width, height) = img.dimensions();

    println!("Loaded texture {path}: {width} x {height}");

    let image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R8G8B8A8_SRGB, // Needs to match image format from device.
            extent: [width, height, 1],
            array_layers: 1,
            usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    )?;

    let buffer: Subbuffer<[u8]> = Buffer::new_slice(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        (width * height * 4) as DeviceSize, // RGBA = 4
    )?;

    {
        let mut writer = buffer.write()?;
        writer.copy_from_slice(img.as_bytes());
    }

    builder.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(buffer, image.clone()))?;

    let image_view = ImageView::new_default(image)?;

    Ok(image_view)
}

fn load_textures(
    models: &[Model],
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: Arc<dyn CommandBufferAllocator>,
    queue: Arc<Queue>,
) -> Result<(Vec<Arc<ImageView>>, HashMap<String, i32>)> /* GLSL int => i32*/ {
    let mut textures = vec![];
    let mut texture_indices: HashMap<String, i32> = HashMap::new();

    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )?;

    for model in models.iter() {
        for path in model.get_texture_paths() {
            if !texture_indices.contains_key(&path) {
                let texture = load_texture(&path, memory_allocator.clone(), &mut builder)?;

                textures.push(texture);
                texture_indices.insert(path.clone(), textures.len() as i32);
            }
        }
    }

    let _ = builder.build()?.execute(queue.clone())?;

    Ok((textures, texture_indices))
}

fn get_material_data(
    prop_type: &MaterialPropertyDataEnum,
    texture_indices: &HashMap<String, i32>,
) -> MaterialPropertyData {
    match prop_type {
        MaterialPropertyDataEnum::None => MaterialPropertyData::default(),
        MaterialPropertyDataEnum::RGB { color } => MaterialPropertyData::new_color(color),
        MaterialPropertyDataEnum::Texture { path } => {
            let texture_index = texture_indices
                .get(path)
                .expect(format!("Texture {path} not found").as_ref());
            MaterialPropertyData::new_texture(*texture_index)
        }
    }
}
