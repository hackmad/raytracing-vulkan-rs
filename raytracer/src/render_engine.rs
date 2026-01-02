use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use anyhow::{Context, Result};
use scene_file::SceneFile;
use shaders::{GfxShaderModules, RtShaderModules, ray_gen};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
        RenderPassBeginInfo, SubpassBeginInfo, SubpassContents, SubpassEndInfo,
    },
    descriptor_set::{DescriptorSet, WriteDescriptorSet},
    format::Format,
    image::{
        Image, ImageAspects, ImageCreateInfo, ImageSubresourceRange, ImageType, ImageUsage,
        SampleCount,
        sampler::{Sampler, SamplerAddressMode, SamplerCreateInfo},
        view::{ImageView, ImageViewCreateInfo, ImageViewType},
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::{PipelineBindPoint, graphics::viewport::Viewport, ray_tracing::ShaderBindingTable},
    render_pass::{Framebuffer, FramebufferCreateInfo},
    sync::GpuFuture,
};

use crate::{
    AnimatedTransform, Camera, Materials, Mesh, MeshInstance, Transform, Vk,
    acceleration::AccelerationStructures,
    create_light_source_alias_table, create_mesh_index_buffer, create_mesh_storage_buffer,
    create_mesh_vertex_buffer,
    pipelines::{GfxPipeline, RtPipeline},
    textures::Textures,
};

#[repr(C)]
#[derive(BufferContents, Clone, Copy)]
pub struct UnifiedPushConstants {
    pub ray_gen_pc: ray_gen::RayGenPushConstants,
}

/// Stores resources specific to the rendering pipelines and renders an image progressively.
/// Each frame renders a batch of samples with a given number of samplers per pixel and accumulates
/// the result over successive calls to its render function.
pub struct RenderEngine {
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

    /// Descriptor set for binding the light source alias table.
    light_source_alias_table_descriptor_set: Arc<DescriptorSet>,

    /// The shader binding table.
    shader_binding_table: ShaderBindingTable,

    /// The raytracing pipeline and layout.
    rt_pipeline: RtPipeline,

    /// The graphics pipeline.
    gfx_pipeline: GfxPipeline,

    /// Combined push constants for all shaders.
    push_constants: UnifiedPushConstants,

    /// Accumulated sample batches.
    accum_image_view: Arc<ImageView>,

    /// Current sample batch to render.
    current_sample_batch: u32,

    /// Number of batches to use when rendering.
    sample_batches: u32,

    /// Acceleration structures. These have to be kept alive since we need the TLAS for rendering.
    _acceleration_structures: AccelerationStructures,
}

impl RenderEngine {
    /// Create vulkano resources for rendering a new scene with given models.
    pub fn new(
        vk: Arc<Vk>,
        scene_file: &SceneFile,
        window_size: &[f32; 2],
        swapchain_format: Format,
    ) -> Result<Self> {
        // Load shader modules.
        let rt_shader_modules = RtShaderModules::load(vk.device.clone());
        let gfx_shader_modules = GfxShaderModules::load(vk.device.clone());

        // Load Textures.
        let textures = Textures::new(vk.clone(), scene_file)?;
        let image_texture_count = textures.image_textures.image_views.len();
        let constant_colour_count = textures.constant_colour_textures.colours.len();
        let checker_texture_count = textures.checker_textures.textures.len();
        let noise_texture_count = textures.noise_textures.textures.len();

        // Get meshes.
        let mut meshes: Vec<Arc<Mesh>> = Vec::new();
        let mut mesh_name_to_index: HashMap<String, usize> = HashMap::new();
        for primitive in scene_file.primitives.iter() {
            let mesh = Arc::new(primitive.into());
            mesh_name_to_index.insert(primitive.get_name().into(), meshes.len());
            meshes.push(mesh);
        }
        let mesh_count = meshes.len();

        // Get instances.
        let mut mesh_instances: Vec<MeshInstance> = Vec::new();
        for instance in scene_file.instances.iter() {
            let mesh_index = mesh_name_to_index
                .get(&instance.name)
                .with_context(|| format!("Mesh {} not found", instance.name))?;

            let mat = instance.get_object_to_world_space_matrix();
            let transform = Transform::Static(AnimatedTransform::from(mat));

            mesh_instances.push(MeshInstance::new(*mesh_index, transform));
        }

        // Get materials.
        let materials = Materials::new(&scene_file.materials, &textures);
        let lambertian_material_count = materials.lambertian_materials.len();
        let metal_material_count = materials.metal_materials.len();
        let dielectric_material_count = materials.dielectric_materials.len();
        let diffuse_light_material_count = materials.diffuse_light_materials.len();

        // Get the light source alias table.
        let light_source_alias_table =
            create_light_source_alias_table(vk.clone(), &mesh_instances, &meshes, &materials)?;

        // Push constants.
        // sampleBatch will need to change in Scene::render() but we can store the push constant
        // data we need for now.
        let push_constants = UnifiedPushConstants {
            ray_gen_pc: ray_gen::RayGenPushConstants {
                resolution: [window_size[0] as u32, window_size[1] as u32],
                samplesPerPixel: scene_file.render.samples_per_pixel,
                sampleBatch: 0,
                maxRayDepth: scene_file.render.max_ray_depth,
                meshCount: mesh_count as _,
                imageTextureCount: image_texture_count as _,
                constantColourCount: constant_colour_count as _,
                checkerTextureCount: checker_texture_count as _,
                noiseTextureCount: noise_texture_count as _,
                lambertianMaterialCount: lambertian_material_count as _,
                metalMaterialCount: metal_material_count as _,
                dielectricMaterialCount: dielectric_material_count as _,
                diffuseLightMaterialCount: diffuse_light_material_count as _,
                lightSourceTriangleCount: light_source_alias_table.triangle_count as _,
                lightSourceTotalArea: light_source_alias_table.total_area as _,
            },
        };

        // Create the graphics pipeline for rendering fullscreen quad.
        let gfx_pipeline = GfxPipeline::new(
            vk.device.clone(),
            &gfx_shader_modules.stages,
            window_size,
            swapchain_format,
        )?;

        // Create the raytracing pipeline.
        let rt_pipeline = RtPipeline::new(
            vk.device.clone(),
            &rt_shader_modules.stages,
            &rt_shader_modules.groups,
            image_texture_count as _,
        )?;
        let pipeline_layout = rt_pipeline.get_layout();
        let layouts = pipeline_layout.set_layouts();

        // Create descriptor sets for non-changing data.

        // Acceleration structures.
        let acceleration_structures =
            AccelerationStructures::new(vk.clone(), &mesh_instances, &meshes)?;

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

        // Light source alias table.
        let light_source_alias_table_descriptor_set = DescriptorSet::new(
            vk.descriptor_set_allocator.clone(),
            layouts[RtPipeline::LIGHT_SOURCE_ALIAS_TABLE].clone(),
            vec![WriteDescriptorSet::buffer(
                0,
                light_source_alias_table.buffer,
            )],
            [],
        )?;

        // Create render image to accumulate sample batches.
        let accum_image_view = create_accumulated_render_image_view(
            vk.clone(),
            window_size[0] as u32,
            window_size[1] as u32,
        )?;

        // Create the shader binding table.
        let shader_binding_table =
            ShaderBindingTable::new(vk.memory_allocator.clone(), &rt_pipeline.get())?;

        Ok(Self {
            tlas_descriptor_set,
            mesh_data_descriptor_set,
            image_textures_descriptor_set,
            constant_colour_textures_descriptor_set,
            other_textures_descriptor_set,
            materials_descriptor_set,
            sky_descriptor_set,
            light_source_alias_table_descriptor_set,
            shader_binding_table,
            rt_pipeline,
            gfx_pipeline,
            push_constants,
            accum_image_view,
            current_sample_batch: 0,
            sample_batches: scene_file.render.sample_batches,
            _acceleration_structures: acceleration_structures,
        })
    }

    /// Updates the resolution for rendering the image.
    pub fn update_image_size(
        &mut self,
        vk: Arc<Vk>,
        image_width: u32,
        image_height: u32,
    ) -> Result<()> {
        // Update resolution for camera.
        self.push_constants.ray_gen_pc.resolution = [image_width, image_height];

        // Update resolution for rendering the accumulated image.
        self.accum_image_view =
            create_accumulated_render_image_view(vk, image_width, image_height)?;

        // Reset the sample batches to restart rendering sample batches again.
        self.current_sample_batch = 0;

        Ok(())
    }

    /// Renders to the given swapchain image view after the given future completes.
    /// This will return a new future for the rendering operation.
    ///
    /// # Panics
    ///
    /// - Panics if render fails for any reason.
    pub fn render(
        &mut self,
        vk: Arc<Vk>,
        before_future: Box<dyn GpuFuture>,
        swapchain_image_view: Arc<ImageView>,
        camera: Arc<RwLock<dyn Camera>>,
    ) -> Box<dyn GpuFuture> {
        // Build a command buffer to bind resources and trace rays.
        let mut builder = AutoCommandBufferBuilder::primary(
            vk.command_buffer_allocator.clone(),
            vk.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        // Perform the rendering passes.
        self.render_raytracing_pass(vk.clone(), camera, &mut builder);
        self.render_graphics_pass(vk.clone(), swapchain_image_view, &mut builder);

        // Build the command buffer.
        let command_buffer = builder.build().unwrap();

        // Execute command buffer.
        let next_future = before_future
            .then_execute(vk.queue.clone(), command_buffer)
            .unwrap();

        next_future.boxed()
    }

    /// Render the next batch of samples using raytracing. If all batches are complete, it returns
    /// early.
    ///
    /// # Panics
    ///
    /// - Panics if render fails for any reason.
    fn render_raytracing_pass(
        &mut self,
        vk: Arc<Vk>,
        camera: Arc<RwLock<dyn Camera>>,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) {
        if self.current_sample_batch >= self.sample_batches {
            return;
        }
        // Create the uniform buffer for the camera.
        let camera = camera.read().unwrap();

        // Create the descriptor sets for the raytracing pipeline.
        let pipeline_layout = self.rt_pipeline.get_layout();
        let layouts = pipeline_layout.set_layouts();

        // Load current sample batch to push constants.
        let mut push_constants = self.push_constants;
        push_constants.ray_gen_pc.sampleBatch = self.current_sample_batch;

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
            [WriteDescriptorSet::image_view(
                0,
                self.accum_image_view.clone(),
            )],
            [],
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
                    self.light_source_alias_table_descriptor_set.clone(),
                ],
            )
            .unwrap()
            .push_constants(pipeline_layout.clone(), 0, push_constants)
            .unwrap()
            .bind_pipeline_ray_tracing(self.rt_pipeline.get())
            .unwrap();

        // https://docs.rs/vulkano/latest/vulkano/shader/index.html#safety
        unsafe {
            builder
                .trace_rays(
                    self.shader_binding_table.addresses().clone(),
                    self.accum_image_view.image().extent(),
                )
                .unwrap();
        }

        // Increment for next batch.
        self.current_sample_batch += 1;
    }

    /// Perform the graphics pass to copy rendered image to the swapchain image view using a
    /// big triangle that covers the viewport.
    ///
    /// It will convert the accumulated sample batches that are linear space to the swapchain
    /// image format which should be sRGB.
    ///
    /// # Panics
    ///
    /// - Panics if render fails for any reason.
    fn render_graphics_pass(
        &mut self,
        vk: Arc<Vk>,
        swapchain_image_view: Arc<ImageView>,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) {
        let extent = swapchain_image_view.image().extent();

        let gfx_pipeline_layout = self.gfx_pipeline.get_layout();
        let gfx_layouts = gfx_pipeline_layout.set_layouts();
        let gfx_render_pass = self.gfx_pipeline.get_render_pass();

        let render_image_sampler =
            Sampler::new(vk.device.clone(), SamplerCreateInfo::simple_repeat_linear()).unwrap();

        let render_image_descriptor_set_2 = DescriptorSet::new(
            vk.descriptor_set_allocator.clone(),
            gfx_layouts[GfxPipeline::RENDER_IMAGE_LAYOUT].clone(),
            [WriteDescriptorSet::image_view_sampler(
                0,
                self.accum_image_view.clone(),
                render_image_sampler,
            )],
            [],
        )
        .unwrap();

        let framebuffer = Framebuffer::new(
            gfx_render_pass,
            FramebufferCreateInfo {
                attachments: vec![swapchain_image_view.clone()],
                ..Default::default()
            },
        )
        .unwrap();

        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into())],
                    ..RenderPassBeginInfo::framebuffer(framebuffer)
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                gfx_pipeline_layout.clone(),
                0,
                vec![render_image_descriptor_set_2],
            )
            .unwrap()
            .bind_pipeline_graphics(self.gfx_pipeline.get())
            .unwrap();

        builder
            .set_viewport(
                0,
                vec![Viewport {
                    offset: [0.0, 0.0],
                    extent: [extent[0] as _, extent[1] as _],
                    depth_range: 0.0..=1.0,
                }]
                .into(),
            )
            .unwrap();

        unsafe { builder.draw(3, 1, 0, 0).unwrap() };

        builder.end_render_pass(SubpassEndInfo::default()).unwrap();
    }
}

/// Create a new image to hold the accumulated sample batches.
fn create_accumulated_render_image_view(
    vk: Arc<Vk>,
    width: u32,
    height: u32,
) -> Result<Arc<ImageView>> {
    let image = Image::new(
        vk.memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R32G32B32A32_SFLOAT,
            extent: [width, height, 1],
            mip_levels: 1,
            array_layers: 1,
            samples: SampleCount::Sample1,
            tiling: vulkano::image::ImageTiling::Optimal,
            usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC | ImageUsage::SAMPLED,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    )?;

    let image_view = ImageView::new(
        image,
        ImageViewCreateInfo {
            view_type: ImageViewType::Dim2d,
            format: Format::R32G32B32A32_SFLOAT,
            subresource_range: ImageSubresourceRange {
                aspects: ImageAspects::COLOR,
                mip_levels: 0..1,
                array_layers: 0..1,
            },
            ..Default::default()
        },
    )?;

    Ok(image_view)
}
