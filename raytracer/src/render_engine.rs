use std::sync::{Arc, RwLock};

use anyhow::Result;
use ash::vk;
use log::debug;
use scene_file::SceneFile;
use shaders::{ClosestHitPushConstants, RayGenPushConstants, UnifiedPushConstants};
use vulkan::{
    Buffer, CommandBuffer, DescriptorSet, DescriptorSetBufferType, Fence, Image, NO_FENCE, Sampler,
    Semaphore, VulkanContext, new_buffer_ds, new_buffers_ds, new_sampler_and_textures_ds,
    new_storage_image_ds, new_tlas_ds,
};

use crate::{
    Camera, Materials, Mesh, RtPipeline, Textures, acceleration::AccelerationStructures,
    create_mesh_index_buffer, create_mesh_storage_buffer, create_mesh_vertex_buffer,
};

struct FrameSyncObjects {
    image_available_semaphore: Semaphore,
    render_finished_semaphore: Semaphore,
    fence: Fence,
}

/// Stores resources specific to the rendering pipeline and renders a frame.
pub struct RenderEngine {
    /// Descriptor set for binding the top-level acceleration structure for the scene.
    tlas_descriptor_set: DescriptorSet<vk::AccelerationStructureKHR>,

    /// Descriptor set for binding mesh data.
    mesh_data_descriptor_set: DescriptorSet<Vec<Buffer>>,

    /// Descriptor set for binding image textures.
    image_textures_descriptor_set: DescriptorSet<Sampler>,

    /// Descriptor set for binding constant colour textures.
    constant_colour_textures_descriptor_set: DescriptorSet<Buffer>,

    /// Descriptor set for binding other textures besides image and constant colour.
    other_textures_descriptor_set: DescriptorSet<Vec<Buffer>>,

    /// Descriptor set for binding materials.
    materials_descriptor_set: DescriptorSet<Vec<Buffer>>,

    /// Descriptor set for binding sky.
    sky_descriptor_set: DescriptorSet<Buffer>,

    /// The raytracing pipeline and layout.
    rt_pipeline: RtPipeline,

    /// Combined push constants for all shaders.
    push_constants: UnifiedPushConstants,

    /// Acceleration structures. These have to be kept alive since we need the TLAS for rendering.
    _acceleration_structures: AccelerationStructures,

    frame_sync_objects: Vec<FrameSyncObjects>,
    current_frame: usize,
}

impl RenderEngine {
    /// Create vulkano resources for rendering a new scene with given models.
    pub fn new(
        context: Arc<VulkanContext>,
        scene_file: &SceneFile,
        window_size: &[f32; 2],
    ) -> Result<Self> {
        // Load Textures.
        let textures = Textures::new(context.clone(), scene_file)?;
        let image_texture_count = textures.image_textures.images.len();
        let constant_colour_count = textures.constant_colour_textures.colours.len();
        let checker_texture_count = textures.checker_textures.textures.len();
        let noise_texture_count = textures.noise_textures.textures.len();

        // Get meshes.
        let meshes: Vec<Mesh> = scene_file.primitives.iter().map(|p| p.into()).collect();

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
            closest_hit_pc: ClosestHitPushConstants {
                mesh_count: meshes.len() as _,
                image_texture_count: image_texture_count as _,
                constant_colour_count: constant_colour_count as _,
                checker_texture_count: checker_texture_count as _,
                noise_texture_count: noise_texture_count as _,
                lambertian_material_count: lambertian_material_count as _,
                metal_material_count: metal_material_count as _,
                dielectric_material_count: dielectric_material_count as _,
                diffuse_light_material_count: diffuse_light_material_count as _,
            },

            ray_gen_pc: RayGenPushConstants {
                resolution: [window_size[0] as u32, window_size[1] as u32],
                samples_per_pixel: scene_file.render.samples_per_pixel,
                sample_batches: scene_file.render.sample_batches,
                sample_batch: 0,
                max_ray_uepth: scene_file.render.max_ray_depth,
            },
        };

        // Create the raytracing pipeline.
        let rt_pipeline = RtPipeline::new(context.clone())?;

        // Create descriptor sets for non-changing data.

        // Acceleration structures.
        let mesh_geometry_buffers = meshes
            .iter()
            .map(|mesh| mesh.create_geometry_buffers(context.clone()))
            .collect::<Result<Vec<_>>>()?;

        let acceleration_structures =
            AccelerationStructures::new(context.clone(), &mesh_geometry_buffers)?;

        // Descriptors.

        let tlas_descriptor_set = new_tlas_ds(
            context.clone(),
            rt_pipeline.set_layouts[RtPipeline::TLAS_LAYOUT],
            acceleration_structures.tlas.acceleration_structure,
        )?;

        // Mesh data.
        // NOTE: The 3 buffers below pack the respective data into a single buffer each where as the
        // acceleration structure is building per-mesh buffes for the different levels of the
        // acceleration structure.
        let vertex_buffer = create_mesh_vertex_buffer(context.clone(), &meshes)?;
        let index_buffer = create_mesh_index_buffer(context.clone(), &meshes)?;
        let mesh_buffer = create_mesh_storage_buffer(context.clone(), &meshes, &materials)?;

        let mesh_data_descriptor_set = new_buffers_ds(
            context.clone(),
            rt_pipeline.set_layouts[RtPipeline::MESH_DATA_LAYOUT],
            DescriptorSetBufferType::Storage,
            vec![vertex_buffer, index_buffer, mesh_buffer],
        )?;

        // Sampler + Textures.
        let texture_sampler = Sampler::new(context.clone())?;

        let texture_image_views = textures
            .image_textures
            .images
            .iter()
            .map(|image| image.image_view);

        let image_textures_descriptor_set = new_sampler_and_textures_ds(
            context.clone(),
            rt_pipeline.set_layouts[RtPipeline::SAMPLERS_AND_TEXTURES_LAYOUT],
            texture_sampler,
            texture_image_views,
        )?;

        // Constant colour textures.
        let constant_colours = if constant_colour_count > 0 {
            textures
                .constant_colour_textures
                .colours
                .iter()
                .map(|&[r, g, b]| [r, g, b, 0.0])
                .collect()
        } else {
            // We cannot create buffer for empty array. Push constants will have material colours count which can
            // be used in shaders to make sure out-of-bounds access can be checked.
            vec![[0.0, 0.0, 0.0, 0.0]]
        };

        let constant_colour_textures_buffer = Buffer::new_device_local_storage_buffer(
            context.clone(),
            vk::BufferUsageFlags::STORAGE_BUFFER,
            &constant_colours,
        )?;

        let constant_colour_textures_descriptor_set = new_buffer_ds(
            context.clone(),
            rt_pipeline.set_layouts[RtPipeline::MATERIAL_COLOURS_LAYOUT],
            DescriptorSetBufferType::Storage,
            constant_colour_textures_buffer,
        )?;

        // Materials.
        let material_buffers = materials.create_buffers(context.clone())?;

        let materials_descriptor_set = new_buffers_ds(
            context.clone(),
            rt_pipeline.set_layouts[RtPipeline::MATERIALS_LAYOUT],
            DescriptorSetBufferType::Storage,
            vec![
                material_buffers.lambertian,
                material_buffers.metal,
                material_buffers.dielectric,
                material_buffers.diffuse_light,
            ],
        )?;

        // Other textures.
        let texture_buffers = textures.create_buffers(context.clone())?;

        let other_textures_descriptor_set = new_buffers_ds(
            context.clone(),
            rt_pipeline.set_layouts[RtPipeline::OTHER_TEXTURES_LAYOUT],
            DescriptorSetBufferType::Storage,
            vec![texture_buffers.checker, texture_buffers.noise],
        )?;

        // Sky.
        debug!("Creating sky uniform buffer");
        let sky_buffer = Buffer::new_device_local_storage_buffer(
            context.clone(),
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            &[scene_file.sky.to_shader()],
        )?;

        let sky_descriptor_set = new_buffer_ds(
            context.clone(),
            rt_pipeline.set_layouts[RtPipeline::SKY_LAYOUT],
            DescriptorSetBufferType::Uniform,
            sky_buffer,
        )?;

        let max_frames_in_flight = context.present_images.len().min(2);
        let mut frame_sync_objects = Vec::with_capacity(max_frames_in_flight);
        for _ in 0..max_frames_in_flight {
            frame_sync_objects.push(FrameSyncObjects {
                image_available_semaphore: Semaphore::new(context.clone())?,
                render_finished_semaphore: Semaphore::new(context.clone())?,
                fence: Fence::new(context.clone(), true)?,
            });
        }

        debug!("Finished setting up render engine");
        Ok(Self {
            tlas_descriptor_set,
            mesh_data_descriptor_set,
            image_textures_descriptor_set,
            constant_colour_textures_descriptor_set,
            other_textures_descriptor_set,
            materials_descriptor_set,
            sky_descriptor_set,
            rt_pipeline,
            push_constants,
            _acceleration_structures: acceleration_structures,
            frame_sync_objects,
            current_frame: 0,
        })
    }

    /// Renders an image view after the given future completes. This will return a new
    /// future for the rendering operation.
    pub fn render(
        &mut self,
        context: Arc<VulkanContext>,
        render_image: &Image,
        camera: Arc<RwLock<dyn Camera>>,
    ) -> Result<()> {
        // Wait for fence to ensure this frame’s work is done.
        let sync = &self.frame_sync_objects[self.current_frame];
        sync.fence.wait_and_reset()?;

        // Create the uniform buffer for the camera.
        let camera = camera.read().unwrap();

        // Create the descriptor sets for the raytracing pipeline.
        let camera = shaders::Camera {
            view_proj: (camera.get_projection_matrix() * camera.get_view_matrix())
                .to_cols_array_2d(),
            view_inverse: camera.get_view_inverse_matrix().to_cols_array_2d(),
            proj_inverse: camera.get_projection_inverse_matrix().to_cols_array_2d(),
            focal_length: camera.get_focal_length(),
            aperture_size: camera.get_aperture_size(),
        };

        debug!("Creating camera buffer");
        let camera_buffer = Buffer::new_device_local_storage_buffer(
            context.clone(),
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            &[camera],
        )
        .unwrap();

        let camera_buffer_descriptor_set = new_buffer_ds(
            context.clone(),
            self.rt_pipeline.set_layouts[RtPipeline::CAMERA_BUFFER_LAYOUT],
            DescriptorSetBufferType::Uniform,
            camera_buffer,
        )
        .unwrap();

        debug!("Creating render render image descriptor set");
        let render_image_descriptor_set = new_storage_image_ds(
            context.clone(),
            self.rt_pipeline.set_layouts[RtPipeline::RENDER_IMAGE_LAYOUT],
            render_image,
        )
        .unwrap();

        // Acquire the swapchain image to render to.
        let (image_index, _) = unsafe {
            context.swapchain_loader.acquire_next_image(
                context.swapchain,
                u64::MAX,
                sync.image_available_semaphore.get(),
                NO_FENCE.get(),
            )?
        };
        let present_image = context.present_images[image_index as usize];
        let present_image_view = context.present_image_views[image_index as usize];
        let present_image_wrapped = Image::new(
            context.clone(),
            present_image,
            present_image_view,
            render_image.width,
            render_image.height,
        );

        // Create a command buffer and record commands to it.
        let command_buffer = CommandBuffer::new(context.clone())?;
        command_buffer.begin_one_time_submit()?;

        // Transition render image to GENERAL
        render_image.transition_layout(
            &command_buffer,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            vk::AccessFlags::empty(),
            vk::AccessFlags::SHADER_WRITE,
        );

        let sample_batches = self.push_constants.ray_gen_pc.sample_batches;
        for sample_batch in 0..sample_batches {
            let mut push_constants = self.push_constants;
            push_constants.ray_gen_pc.sample_batch = sample_batch as _;

            self.rt_pipeline.record_commands(
                &command_buffer,
                &[
                    self.tlas_descriptor_set.set,
                    camera_buffer_descriptor_set.set,
                    render_image_descriptor_set.set,
                    self.mesh_data_descriptor_set.set,
                    self.image_textures_descriptor_set.set,
                    self.constant_colour_textures_descriptor_set.set,
                    self.materials_descriptor_set.set,
                    self.other_textures_descriptor_set.set,
                    self.sky_descriptor_set.set,
                ],
                &push_constants,
            );
        }

        // Transition render image for transfer.
        render_image.transition_layout(
            &command_buffer,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
        );

        // Transition swapchain image to transfer dst
        present_image_wrapped.transition_layout(
            &command_buffer,
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
        );

        // Blit render image → swapchain image.
        command_buffer.blit_image(
            render_image.image,
            present_image_wrapped.image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::Extent3D {
                width: render_image.width,
                height: render_image.height,
                depth: 1,
            },
            vk::Extent3D {
                width: render_image.width,
                height: render_image.height,
                depth: 1,
            },
            vk::Filter::NEAREST,
        );

        // Transition swapchain image to present.
        present_image_wrapped.transition_layout(
            &command_buffer,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::empty(),
        );

        // End command buffer.
        command_buffer.end()?;

        // Submit the command buffer, signaling semaphores and waiting on fences.
        let image_available_semaphores = [sync.image_available_semaphore.get()];
        let render_finished_semaphores = [sync.render_finished_semaphore.get()];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&image_available_semaphores)
            .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
            .signal_semaphores(&render_finished_semaphores);

        command_buffer.submit(Some(submit_info), &sync.fence)?;

        unsafe {
            let status = context.device.get_fence_status(sync.fence.get());
            debug!("Fence status: {status:?}");
        }

        // Present.
        debug!("Presenting swapchain image");
        let swapchains = [context.swapchain];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&render_finished_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        unsafe {
            context
                .swapchain_loader
                .queue_present(context.present_queue, &present_info)?;
        }

        // Advance current frame index (wrap around).
        self.current_frame = (self.current_frame + 1) % self.frame_sync_objects.len();

        Ok(())
    }
}
