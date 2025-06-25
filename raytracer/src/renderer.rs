use std::sync::{Arc, RwLock};

use anyhow::Result;
use scene_file::SceneFile;
use shaders::{ShaderModules, closest_hit, ray_gen};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage},
    descriptor_set::{DescriptorSet, WriteDescriptorSet},
    image::{
        sampler::{Sampler, SamplerAddressMode, SamplerCreateInfo},
        view::ImageView,
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::{PipelineBindPoint, ray_tracing::ShaderBindingTable},
    sync::GpuFuture,
};

use crate::{
    Camera, Materials, Vk, acceleration::AccelerationStructures, create_mesh_index_buffer,
    create_mesh_storage_buffer, create_mesh_vertex_buffer, pipeline::RtPipeline,
    textures::Textures,
};

#[repr(C)]
#[derive(BufferContents, Clone, Copy)]
pub struct UnifiedPushConstants {
    // RayGen: 0–23
    pub ray_gen_pc: ray_gen::RayGenPushConstants,

    // ClosestHit: 24–55
    pub closest_hit_pc: closest_hit::ClosestHitPushConstants,
}

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

    /// Descriptor set for binding other textures besides image and constant colour.
    other_textures_descriptor_set: Arc<DescriptorSet>,

    /// Descriptor set for binding materials.
    materials_descriptor_set: Arc<DescriptorSet>,

    /// Descriptor set for binding sky.
    sky_descriptor_set: Arc<DescriptorSet>,

    /// The shader binding table.
    shader_binding_table: ShaderBindingTable,

    /// The raytracing pipeline and layout.
    rt_pipeline: RtPipeline,

    /// Combined push constants for all shaders.
    push_constants: UnifiedPushConstants,

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
        let noise_texture_count = textures.noise_textures.textures.len();

        // Get meshes.
        let meshes: Vec<_> = scene_file.primitives.iter().map(|o| o.into()).collect();

        // Get materials.
        let materials = Materials::new(&scene_file.materials, &textures);
        let lambertian_material_count = materials.lambertian_materials.len();
        let metal_material_count = materials.metal_materials.len();
        let dielectric_material_count = materials.dielectric_materials.len();
        let diffuse_light_material_count = materials.diffuse_light_materials.len();

        // Push constants.
        // sampleBatch will need to change in Scene::render() but we can store the push constant
        // data we need for now.
        let push_constants = UnifiedPushConstants {
            closest_hit_pc: closest_hit::ClosestHitPushConstants {
                meshCount: meshes.len() as _,
                imageTextureCount: image_texture_count as _,
                constantColourCount: constant_colour_count as _,
                checkerTextureCount: checker_texture_count as _,
                noiseTextureCount: noise_texture_count as _,
                lambertianMaterialCount: lambertian_material_count as _,
                metalMaterialCount: metal_material_count as _,
                dielectricMaterialCount: dielectric_material_count as _,
                diffuseLightMaterialCount: diffuse_light_material_count as _,
            },

            ray_gen_pc: ray_gen::RayGenPushConstants {
                resolution: [window_size[0] as u32, window_size[1] as u32],
                samplesPerPixel: scene_file.render.samples_per_pixel,
                sampleBatches: scene_file.render.sample_batches,
                sampleBatch: 0,
                maxRayDepth: scene_file.render.max_ray_depth,
            },
        };

        // Create the raytracing pipeline.
        let rt_pipeline = RtPipeline::new(
            vk.device.clone(),
            &shader_modules.stages,
            &shader_modules.groups,
            image_texture_count as _,
        )?;
        let pipeline_layout = rt_pipeline.get_layout();
        let layouts = pipeline_layout.set_layouts();

        // Create descriptor sets for non-changing data.

        // Acceleration structures.
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

        // Mesh data.
        let vertex_buffer = create_mesh_vertex_buffer(vk.clone(), &meshes)?;
        let index_buffer = create_mesh_index_buffer(vk.clone(), &meshes)?;
        let mesh_buffer = create_mesh_storage_buffer(vk.clone(), &meshes, &materials)?;

        let mesh_data_descriptor_set = DescriptorSet::new(
            vk.descriptor_set_allocator.clone(),
            layouts[RtPipeline::MESH_DATA_LAYOUT].clone(),
            [
                WriteDescriptorSet::buffer(0, vertex_buffer),
                WriteDescriptorSet::buffer(1, index_buffer),
                WriteDescriptorSet::buffer(2, mesh_buffer),
            ],
            [],
        )?;

        // Sampler + Textures.
        let sampler = Sampler::new(
            vk.device.clone(),
            SamplerCreateInfo {
                address_mode: [SamplerAddressMode::Repeat; 3],
                ..Default::default()
            },
        )?;

        let mut image_texture_descriptor_writes = vec![WriteDescriptorSet::sampler(0, sampler)];

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

        // Constant colour textures.
        let constant_colours = if constant_colour_count > 0 {
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
            constant_colours,
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
                WriteDescriptorSet::buffer(3, material_buffers.diffuse_light),
            ],
            [],
        )?;

        // Other textures.
        let texture_buffers = textures.create_buffers(vk.clone())?;

        let other_textures_descriptor_set = DescriptorSet::new(
            vk.descriptor_set_allocator.clone(),
            layouts[RtPipeline::OTHER_TEXTURES_LAYOUT].clone(),
            vec![
                WriteDescriptorSet::buffer(0, texture_buffers.checker),
                WriteDescriptorSet::buffer(1, texture_buffers.noise),
            ],
            [],
        )?;

        // Sky.
        let sky_buffer = Buffer::from_data(
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
            scene_file.sky.to_shader(),
        )?;
        let sky_descriptor_set = DescriptorSet::new(
            vk.descriptor_set_allocator.clone(),
            layouts[RtPipeline::SKY_LAYOUT].clone(),
            vec![WriteDescriptorSet::buffer(0, sky_buffer)],
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
            other_textures_descriptor_set,
            materials_descriptor_set,
            sky_descriptor_set,
            shader_binding_table,
            rt_pipeline,
            push_constants,
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

        let sample_batches = self.push_constants.ray_gen_pc.sampleBatches;
        for sample_batch in 0..sample_batches {
            let mut push_constants = self.push_constants;
            push_constants.ray_gen_pc.sampleBatch = sample_batch as _;

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
                        self.other_textures_descriptor_set.clone(),
                        self.sky_descriptor_set.clone(),
                    ],
                )
                .unwrap()
                .push_constants(pipeline_layout.clone(), 0, self.push_constants)
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
