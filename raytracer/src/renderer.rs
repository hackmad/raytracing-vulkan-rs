use std::sync::{Arc, RwLock};

use anyhow::Result;
use log::debug;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
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

use crate::{
    Camera, Materials, SceneFile, Vk,
    acceleration::AccelerationStructures,
    create_mesh_storage_buffer,
    pipeline::RtPipeline,
    shaders::{ShaderModules, closest_hit, ray_gen},
    textures::Textures,
};

/// Stores resources specific to the rendering pipeline and renders a frame.
pub struct Renderer {
    /// Descriptor set for binding the top-level acceleration structure for the scene.
    tlas_descriptor_set: Arc<DescriptorSet>,

    /// Descriptor set for binding mesh data.
    mesh_data_descriptor_set: Arc<DescriptorSet>,

    /// Descriptor set for binding image textures.
    image_textures_descriptor_set: Arc<DescriptorSet>,

    /// Descriptor set for binding constant colour textures.
    constant_colour_textures_descriptor_set: Arc<DescriptorSet>,

    /// Descriptor set for binding checker textures.
    checker_textures_descriptor_set: Arc<DescriptorSet>,

    /// Descriptor set for binding materials.
    materials_descriptor_set: Arc<DescriptorSet>,

    /// The shader binding table.
    shader_binding_table: ShaderBindingTable,

    /// The raytracing pipeline and layout.
    rt_pipeline: RtPipeline,

    /// Push constants for the closest hit shader.
    closest_hit_push_constants: closest_hit::ClosestHitPushConstants,

    /// Push constants for the ray generation shader.
    ray_gen_push_constants: ray_gen::RayGenPushConstants,

    /// Acceleration structures. These have to be kept alive since we need the TLAS for rendering.
    _acceleration_structures: AccelerationStructures,
}

impl Renderer {
    /// Create vulkano resources for rendering a new scene with given models.
    pub fn new(vk: Arc<Vk>, scene_file: &SceneFile, window_size: &[f32; 2]) -> Result<Self> {
        // Load shader modules.
        let shader_modules = ShaderModules::load(vk.device.clone());

        // Load Textures.
        let textures = Textures::new(vk.clone(), scene_file)?;
        let image_texture_count = textures.image_textures.image_views.len();
        let constant_colour_count = textures.constant_colour_textures.colours.len();
        let checker_texture_count = textures.checker_textures.textures.len();

        // Get meshes.
        let meshes = scene_file.get_meshes();

        // Get materials.
        let materials = Materials::new(&scene_file.materials, &textures);
        debug!("{materials:?}");

        // Push constants.
        let closest_hit_push_constants = closest_hit::ClosestHitPushConstants {
            imageTextureCount: image_texture_count as _,
            constantColourCount: constant_colour_count as _,
            checkerTextureCount: checker_texture_count as _,
            lambertianMaterialCount: materials.lambertian_materials.len() as _,
            metalMaterialCount: materials.metal_materials.len() as _,
            dielectricMaterialCount: materials.dielectric_materials.len() as _,
        };

        // sampleBatch will need to change in Scene::render() but we can store the push constant
        // data we need for now.
        let ray_gen_push_constants = ray_gen::RayGenPushConstants {
            resolution: [window_size[0] as u32, window_size[1] as u32],
            samplesPerPixel: scene_file.render.samples_per_pixel,
            sampleBatches: scene_file.render.sample_batches,
            sampleBatch: 0,
            maxRayDepth: scene_file.render.max_ray_depth,
        };

        // Create the raytracing pipeline.
        let rt_pipeline = RtPipeline::new(
            vk.device.clone(),
            &shader_modules.stages,
            &shader_modules.groups,
            image_texture_count as _,
            size_of::<closest_hit::ClosestHitPushConstants>() as _,
            size_of::<ray_gen::RayGenPushConstants>() as _,
        )?;
        let pipeline_layout = rt_pipeline.get_layout();
        let layouts = pipeline_layout.set_layouts();

        // For now the acceleration structure is non-changing. We can create its descriptor set
        // and clone it later during render.
        let acceleration_structures = AccelerationStructures::new(vk.clone(), &meshes)?;

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
        let mesh_buffer = create_mesh_storage_buffer(vk.clone(), &meshes, &materials)?;

        let mesh_data_descriptor_set = DescriptorSet::new(
            vk.descriptor_set_allocator.clone(),
            layouts[RtPipeline::MESH_DATA_LAYOUT].clone(),
            [WriteDescriptorSet::buffer(0, mesh_buffer)],
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

        let mut image_texture_descriptor_writes =
            vec![WriteDescriptorSet::sampler(0, sampler.clone())];
        if image_texture_count > 0 {
            // We cannot create descriptor set for empty array. Push constants will have texture count which can
            // be used in shaders to make sure out-of-bounds access can be checked.
            image_texture_descriptor_writes.push(WriteDescriptorSet::image_view_array(
                1,
                0,
                textures.image_textures.image_views.clone(),
            ));
        }

        let image_textures_descriptor_set = DescriptorSet::new_variable(
            vk.descriptor_set_allocator.clone(),
            layouts[RtPipeline::SAMPLERS_AND_TEXTURES_LAYOUT].clone(),
            image_texture_count as _,
            image_texture_descriptor_writes,
            [],
        )?;

        // Material colours
        let mat_colours = if constant_colour_count > 0 {
            textures.constant_colour_textures.colours.clone()
        } else {
            // We cannot create buffer for empty array. Push constants will have material colours count which can
            // be used in shaders to make sure out-of-bounds access can be checked.
            vec![[0.0, 0.0, 0.0]]
        };

        let constant_colour_textures_buffer = Buffer::from_iter(
            vk.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            mat_colours,
        )?;

        let constant_colour_textures_descriptor_set = DescriptorSet::new(
            vk.descriptor_set_allocator.clone(),
            layouts[RtPipeline::MATERIAL_COLOURS_LAYOUT].clone(),
            vec![WriteDescriptorSet::buffer(
                0,
                constant_colour_textures_buffer,
            )],
            [],
        )?;

        // Materials.
        let material_buffers = materials.create_buffers(vk.clone())?;

        let materials_descriptor_set = DescriptorSet::new(
            vk.descriptor_set_allocator.clone(),
            layouts[RtPipeline::MATERIALS_LAYOUT].clone(),
            vec![
                WriteDescriptorSet::buffer(0, material_buffers.lambertian),
                WriteDescriptorSet::buffer(1, material_buffers.metal),
                WriteDescriptorSet::buffer(2, material_buffers.dielectric),
            ],
            [],
        )?;

        // Check textures.
        let texture_buffers = textures.create_buffers(vk.clone())?;

        let checker_textures_descriptor_set = DescriptorSet::new(
            vk.descriptor_set_allocator.clone(),
            layouts[RtPipeline::CHECKER_TEXTURES_LAYOUT].clone(),
            vec![WriteDescriptorSet::buffer(0, texture_buffers.checker)],
            [],
        )?;

        // Create the shader binding table.
        let shader_binding_table =
            ShaderBindingTable::new(vk.memory_allocator.clone(), &rt_pipeline.get())?;

        Ok(Renderer {
            tlas_descriptor_set,
            mesh_data_descriptor_set,
            image_textures_descriptor_set,
            constant_colour_textures_descriptor_set,
            checker_textures_descriptor_set,
            materials_descriptor_set,
            shader_binding_table,
            rt_pipeline,
            closest_hit_push_constants,
            ray_gen_push_constants,
            _acceleration_structures: acceleration_structures,
        })
    }

    /// Renders an image view after the given future completes. This will return a new
    /// future for the rendering operation.
    ///
    /// # Panics
    ///
    /// - Panics if any Vulkan resources fail to create.
    pub fn render(
        &mut self,
        vk: Arc<Vk>,
        before_future: Box<dyn GpuFuture>,
        image_view: Arc<ImageView>,
        camera: Arc<RwLock<dyn Camera>>,
    ) -> Box<dyn GpuFuture> {
        // Create the uniform buffer for the camera.
        let camera = camera.read().unwrap();

        // Create the descriptor sets for the raytracing pipeline.
        let pipeline_layout = self.rt_pipeline.get_layout();
        let layouts = pipeline_layout.set_layouts();

        let mut future = before_future;

        let sample_batches = self.ray_gen_push_constants.sampleBatches;
        for sample_batch in 0..sample_batches {
            let mut ray_gen_push_constants = self.ray_gen_push_constants;
            ray_gen_push_constants.sampleBatch = sample_batch as _;

            let camera_buffer = Buffer::from_data(
                vk.memory_allocator.clone(),
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
                    focalLength: camera.get_focal_length(),
                    apertureSize: camera.get_aperture_size(),
                },
            )
            .unwrap();

            let camera_buffer_descriptor_set = DescriptorSet::new(
                vk.descriptor_set_allocator.clone(),
                layouts[RtPipeline::CAMERA_BUFFER_LAYOUT].clone(),
                [WriteDescriptorSet::buffer(0, camera_buffer)],
                [],
            )
            .unwrap();

            let render_image_descriptor_set = DescriptorSet::new(
                vk.descriptor_set_allocator.clone(),
                layouts[RtPipeline::RENDER_IMAGE_LAYOUT].clone(),
                [WriteDescriptorSet::image_view(0, image_view.clone())],
                [],
            )
            .unwrap();

            // Build a command buffer to bind resources and trace rays.
            let mut builder = AutoCommandBufferBuilder::primary(
                vk.command_buffer_allocator.clone(),
                vk.queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

            builder
                .bind_descriptor_sets(
                    PipelineBindPoint::RayTracing,
                    pipeline_layout.clone(),
                    0,
                    vec![
                        self.tlas_descriptor_set.clone(),
                        camera_buffer_descriptor_set,
                        render_image_descriptor_set,
                        self.mesh_data_descriptor_set.clone(),
                        self.image_textures_descriptor_set.clone(),
                        self.constant_colour_textures_descriptor_set.clone(),
                        self.materials_descriptor_set.clone(),
                        self.checker_textures_descriptor_set.clone(),
                    ],
                )
                .unwrap()
                .push_constants(pipeline_layout.clone(), 0, self.closest_hit_push_constants)
                .unwrap()
                .push_constants(pipeline_layout.clone(), 16, ray_gen_push_constants)
                .unwrap()
                .bind_pipeline_ray_tracing(self.rt_pipeline.get())
                .unwrap();

            // https://docs.rs/vulkano/latest/vulkano/shader/index.html#safety
            unsafe {
                builder
                    .trace_rays(
                        self.shader_binding_table.addresses().clone(),
                        image_view.image().extent(),
                    )
                    .unwrap();
            }

            let command_buffer = builder.build().unwrap();

            let next_future = future
                .then_execute(vk.queue.clone(), command_buffer)
                .unwrap();

            future = next_future.boxed();
        }

        future
    }
}
