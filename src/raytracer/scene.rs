use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, allocator::CommandBufferAllocator,
    },
    descriptor_set::{DescriptorSet, WriteDescriptorSet, allocator::DescriptorSetAllocator},
    device::{Device, Queue},
    image::{
        sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
        view::ImageView,
    },
    memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter},
    pipeline::{
        PipelineBindPoint, PipelineShaderStageCreateInfo,
        ray_tracing::{RayTracingShaderGroupCreateInfo, ShaderBindingTable},
    },
    sync::GpuFuture,
};

use super::{
    Camera, MaterialPropertyData, MaterialPropertyDataEnum,
    acceleration::AccelerationStructures,
    model::Model,
    pipeline::RtPipeline,
    shaders::{ShaderModules, closest_hit, ray_gen},
    texture::Textures,
};

pub struct Scene {
    queue: Arc<Queue>,
    descriptor_set_allocator: Arc<dyn DescriptorSetAllocator>,
    tlas_descriptor_set: Arc<DescriptorSet>,
    mesh_data_descriptor_set: Arc<DescriptorSet>,
    textures_descriptor_set: Arc<DescriptorSet>,
    shader_binding_table: ShaderBindingTable,
    rt_pipeline: RtPipeline,
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: Arc<dyn CommandBufferAllocator>,

    /// Acceleration structures. These have to be kept alive since we need the TLAS for rendering.
    _acceleration_structures: AccelerationStructures,

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
        let textures = Textures::load(
            models,
            memory_allocator.clone(),
            command_buffer_allocator.clone(),
            queue.clone(),
        )
        .unwrap();

        // Create the raytracing pipeline.
        let rt_pipeline = RtPipeline::new(
            device.clone(),
            &stages,
            &groups,
            textures.image_views.len() as _,
        )
        .unwrap();
        let pipeline_layout = rt_pipeline.get_layout();
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
            layouts[RtPipeline::TLAS_LAYOUT].clone(),
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
            layouts[RtPipeline::MESH_DATA_LAYOUT].clone(),
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
            layouts[RtPipeline::SAMPLERS_AND_TEXTURES_LAYOUT].clone(),
            textures.image_views.len() as _,
            [
                WriteDescriptorSet::sampler(0, sampler.clone()),
                WriteDescriptorSet::image_view_array(1, 0, textures.image_views),
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
            ShaderBindingTable::new(memory_allocator.clone(), &rt_pipeline.get()).unwrap();

        Scene {
            queue,
            descriptor_set_allocator,
            tlas_descriptor_set,
            mesh_data_descriptor_set,
            textures_descriptor_set,
            shader_binding_table,
            rt_pipeline,
            memory_allocator,
            command_buffer_allocator,
            _acceleration_structures: acceleration_structures,
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

        let pipeline_layout = self.rt_pipeline.get_layout();
        let layouts = pipeline_layout.set_layouts();

        let uniform_buffer_descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            layouts[RtPipeline::UNIFORM_BUFFER_LAYOUT].clone(),
            [WriteDescriptorSet::buffer(0, uniform_buffer)],
            [],
        )
        .unwrap();

        let render_image_descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            layouts[RtPipeline::RENDER_IMAGE_LAYOUT].clone(),
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
                self.rt_pipeline.get_layout(),
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
            .bind_pipeline_ray_tracing(self.rt_pipeline.get())
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
