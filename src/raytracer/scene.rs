use super::{
    Camera, LightPropertyData, Vk,
    acceleration::AccelerationStructures,
    create_device_local_buffer, create_mesh_storage_buffer,
    model::Model,
    pipeline::RtPipeline,
    shaders::{ShaderModules, closest_hit, ray_gen},
    texture::Textures,
};
use crate::raytracer::MaterialColours;
use anyhow::Result;
use std::sync::{Arc, RwLock};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage},
    descriptor_set::{DescriptorSet, WriteDescriptorSet},
    image::{
        sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
        view::ImageView,
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::{PipelineBindPoint, ray_tracing::ShaderBindingTable},
    sync::GpuFuture,
};

/// The vulkano resources specific to the rendering pipeline.
struct SceneResources {
    /// Descriptor set for binding the top-level acceleration structure for the scene.
    tlas_descriptor_set: Arc<DescriptorSet>,

    /// Descriptor set for binding mesh data.
    mesh_data_descriptor_set: Arc<DescriptorSet>,

    /// Descriptor set for binding textures.
    textures_descriptor_set: Arc<DescriptorSet>,

    /// Descriptor set for binding material data.
    scene_data_descriptor_set: Arc<DescriptorSet>,

    /// The shader binding table.
    shader_binding_table: ShaderBindingTable,

    /// The raytracing pipeline and layout.
    rt_pipeline: RtPipeline,

    /// Push constants for the closest hit shader.
    closest_hit_push_constants: closest_hit::PushConstantData,

    /// Push constants for the ray generation shader.
    ray_gen_push_constants: ray_gen::PushConstantData,

    /// Acceleration structures. These have to be kept alive since we need the TLAS for rendering.
    _acceleration_structures: AccelerationStructures,
}

impl SceneResources {
    /// Create vulkano resources for rendering a new scene with given models.
    fn new(
        vk: Arc<Vk>,
        models: &[Model],
        lights: &[LightPropertyData],
        window_size: [f32; 2],
    ) -> Result<Self> {
        // Load shader modules.
        let shader_modules = ShaderModules::load(vk.device.clone());

        // Load Textures.
        let textures = Textures::load(models, vk.clone())?;
        let texture_count = textures.image_views.len() as u32;
        //println!("{textures:?}");

        // Load material colours.
        let material_colours = MaterialColours::load(models);
        let material_colour_count = material_colours.colours.len() as u32;
        //println!("{material_colours:?}");

        // Push constants.
        let closest_hit_push_constants = closest_hit::PushConstantData {
            textureCount: texture_count,
            materialColourCount: material_colour_count,
            lightCount: (lights.len() as u32).into(),
        };
        let closest_hit_push_constants_bytes = size_of::<closest_hit::PushConstantData>() as u32;

        let ray_gen_push_constants = ray_gen::PushConstantData {
            resolution: [window_size[0] as u32, window_size[1] as u32],
            samplesPerPixel: 100,
            maxRayDepth: 50,
        };
        let ray_gen_push_constants_bytes = size_of::<ray_gen::PushConstantData>() as u32;

        // Create the raytracing pipeline.
        let rt_pipeline = RtPipeline::new(
            vk.device.clone(),
            &shader_modules.stages,
            &shader_modules.groups,
            texture_count,
            closest_hit_push_constants_bytes,
            ray_gen_push_constants_bytes,
        )?;
        let pipeline_layout = rt_pipeline.get_layout();
        let layouts = pipeline_layout.set_layouts();

        // For now the acceleration structure is non-changing. We can create its descriptor set
        // and clone it later during render.
        let acceleration_structures = AccelerationStructures::new(vk.clone(), models)?;

        let tlas_descriptor_set = DescriptorSet::new(
            vk.descriptor_set_allocator.clone(),
            layouts[RtPipeline::TLAS_LAYOUT].clone(),
            [WriteDescriptorSet::acceleration_structure(
                0,
                acceleration_structures.tlas.clone(),
            )],
            [],
        )?;

        // Mesh data won't change either. We can create its descriptor set and clone it later
        // during render.
        let mesh_data =
            create_mesh_storage_buffer(vk.clone(), models, &textures, &material_colours)?;

        let mesh_data_descriptor_set = DescriptorSet::new(
            vk.descriptor_set_allocator.clone(),
            layouts[RtPipeline::MESH_DATA_LAYOUT].clone(),
            [WriteDescriptorSet::buffer(0, mesh_data)],
            [],
        )?;

        // Textures + Sampler
        let sampler = Sampler::new(
            vk.device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                ..Default::default()
            },
        )?;

        let mut texture_descriptor_writes = vec![WriteDescriptorSet::sampler(0, sampler.clone())];
        if texture_count > 0 {
            // We cannot create descriptor set for empty array. Push constants will have texture count which can
            // be used in shaders to make sure out-of-bounds access can be checked.
            texture_descriptor_writes.push(WriteDescriptorSet::image_view_array(
                1,
                0,
                textures.image_views,
            ));
        }

        let textures_descriptor_set = DescriptorSet::new_variable(
            vk.descriptor_set_allocator.clone(),
            layouts[RtPipeline::SAMPLERS_AND_TEXTURES_LAYOUT].clone(),
            texture_count as _,
            texture_descriptor_writes,
            [],
        )?;

        // Scene data.
        let mat_colours = if material_colour_count > 0 {
            material_colours.colours
        } else {
            // We cannot create buffer for empty array. Push constants will have material colours count which can
            // be used in shaders to make sure out-of-bounds access can be checked.
            vec![[0.0, 0.0, 0.0]]
        };

        let material_colours_buffer = create_device_local_buffer(
            vk.clone(),
            BufferUsage::STORAGE_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
            mat_colours,
        )?;

        let lights_buffer: Subbuffer<[closest_hit::Light]> = create_device_local_buffer(
            vk.clone(),
            BufferUsage::STORAGE_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
            lights.iter().map(|light| light.into()),
        )?;

        let scene_data_descriptor_set = DescriptorSet::new(
            vk.descriptor_set_allocator.clone(),
            layouts[RtPipeline::SCENE_DATA_LAYOUT].clone(),
            vec![
                WriteDescriptorSet::buffer(
                    RtPipeline::MATERIAL_COLOURS_BINDING_INDEX,
                    material_colours_buffer,
                ),
                WriteDescriptorSet::buffer(RtPipeline::LIGHTS_BINDING_INDEX, lights_buffer),
            ],
            [],
        )?;

        // Create the shader binding table.
        let shader_binding_table =
            ShaderBindingTable::new(vk.memory_allocator.clone(), &rt_pipeline.get())?;

        Ok(SceneResources {
            tlas_descriptor_set,
            mesh_data_descriptor_set,
            textures_descriptor_set,
            scene_data_descriptor_set,
            shader_binding_table,
            rt_pipeline,
            closest_hit_push_constants,
            ray_gen_push_constants,
            _acceleration_structures: acceleration_structures,
        })
    }
}

/// Describes the scene for raytracing.
pub struct Scene {
    /// Vulkano conext.
    vk: Arc<Vk>,

    /// Camera.
    camera: Arc<RwLock<dyn Camera>>,

    /// Lights.
    lights: Vec<LightPropertyData>,

    /// Vulkano resources specific to the rendering pipeline.
    resources: Option<SceneResources>,
}

impl Scene {
    /// Create a new scene from the given models and camera.
    pub fn new(
        vk: Arc<Vk>,
        models: &[Model],
        camera: Arc<RwLock<dyn Camera>>,
        lights: &[LightPropertyData],
        window_size: [f32; 2],
    ) -> Result<Self> {
        if models.len() == 0 {
            Ok(Scene {
                vk,
                resources: None,
                camera,
                lights: Vec::from(lights),
            })
        } else {
            SceneResources::new(vk.clone(), models, lights, window_size).map(|resources| Scene {
                vk,
                resources: Some(resources),
                camera,
                lights: Vec::from(lights),
            })
        }
    }

    /// Rebuilds the scene with new models.
    pub fn rebuild(&mut self, models: &[Model], window_size: [f32; 2]) -> Result<()> {
        let resources = SceneResources::new(self.vk.clone(), models, &self.lights, window_size)?;
        self.resources = Some(resources);
        Ok(())
    }

    /// Updates the camera image size to match a new window size.
    pub fn update_window_size(&mut self, window_size: [f32; 2]) {
        let mut camera = self.camera.write().unwrap();
        camera.update_image_size(window_size[0] as u32, window_size[1] as u32);
    }

    /// Renders a scene to an image view after the given future completes. This will return a new
    /// future for the rendering operation.
    ///
    /// # Panics
    ///
    /// - Panics if any Vulkan resources fail to create.
    pub fn render(
        &self,
        before_future: Box<dyn GpuFuture>,
        image_view: Arc<ImageView>,
    ) -> Box<dyn GpuFuture> {
        if let Some(resources) = self.resources.as_ref() {
            // Create the uniform buffer for the camera.
            let camera = self.camera.read().unwrap();

            let uniform_buffer = Buffer::from_data(
                self.vk.memory_allocator.clone(),
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

            // Create the descriptor sets for the raytracing pipeline.
            let pipeline_layout = resources.rt_pipeline.get_layout();
            let layouts = pipeline_layout.set_layouts();

            let uniform_buffer_descriptor_set = DescriptorSet::new(
                self.vk.descriptor_set_allocator.clone(),
                layouts[RtPipeline::UNIFORM_BUFFER_LAYOUT].clone(),
                [WriteDescriptorSet::buffer(0, uniform_buffer)],
                [],
            )
            .unwrap();

            let render_image_descriptor_set = DescriptorSet::new(
                self.vk.descriptor_set_allocator.clone(),
                layouts[RtPipeline::RENDER_IMAGE_LAYOUT].clone(),
                [WriteDescriptorSet::image_view(0, image_view.clone())],
                [],
            )
            .unwrap();

            // Build a command buffer to bind resources and trace rays.
            let mut builder = AutoCommandBufferBuilder::primary(
                self.vk.command_buffer_allocator.clone(),
                self.vk.queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

            builder
                .bind_descriptor_sets(
                    PipelineBindPoint::RayTracing,
                    resources.rt_pipeline.get_layout(),
                    0,
                    vec![
                        resources.tlas_descriptor_set.clone(),
                        uniform_buffer_descriptor_set,
                        render_image_descriptor_set,
                        resources.mesh_data_descriptor_set.clone(),
                        resources.textures_descriptor_set.clone(),
                        resources.scene_data_descriptor_set.clone(),
                    ],
                )
                .unwrap()
                .push_constants(
                    resources.rt_pipeline.get_layout(),
                    0,
                    resources.closest_hit_push_constants.clone(),
                )
                .unwrap()
                .push_constants(
                    resources.rt_pipeline.get_layout(),
                    0,
                    resources.ray_gen_push_constants.clone(),
                )
                .unwrap()
                .bind_pipeline_ray_tracing(resources.rt_pipeline.get())
                .unwrap();

            // https://docs.rs/vulkano/latest/vulkano/shader/index.html#safety
            unsafe {
                builder
                    .trace_rays(
                        resources.shader_binding_table.addresses().clone(),
                        image_view.image().extent(),
                    )
                    .unwrap();
            }

            let command_buffer = builder.build().unwrap();

            let after_future = before_future
                .then_execute(self.vk.queue.clone(), command_buffer)
                .unwrap();

            after_future.boxed()
        } else {
            // Do nothing.
            let after_future = before_future;
            after_future.boxed()
        }
    }
}
