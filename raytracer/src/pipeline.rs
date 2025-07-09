use std::sync::Arc;

use anyhow::{Result, anyhow};
use ash::{khr, vk};
use log::debug;
use shaders::{ClosestHitPushConstants, RayGenPushConstants, ShaderModules, UnifiedPushConstants};
use vulkan::{Buffer, CommandBuffer, DescriptorSetLayout, VulkanContext};

const ENTRY_POINT: &core::ffi::CStr = c"main";

/// The raytracing pipeline.
pub struct RtPipeline {
    context: Arc<VulkanContext>,
    rt_loader: khr::ray_tracing_pipeline::Device,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    pub set_layouts: Vec<DescriptorSetLayout>,

    sbt_ray_gen_region: vk::StridedDeviceAddressRegionKHR,
    sbt_ray_miss_region: vk::StridedDeviceAddressRegionKHR,
    sbt_closest_hit_region: vk::StridedDeviceAddressRegionKHR,
    sbt_call_region: vk::StridedDeviceAddressRegionKHR,
    _sbt_buffer: Buffer,
}

impl RtPipeline {
    // These make it easier to set the descriptor set layout.

    /// Top-level acceleration structure.
    pub const TLAS_LAYOUT: usize = 0;

    /// Uniform buffer for the camera data.
    pub const CAMERA_BUFFER_LAYOUT: usize = 1;

    /// Storage image used for rendering.
    pub const RENDER_IMAGE_LAYOUT: usize = 2;

    /// Storage buffer used for mesh data.
    pub const MESH_DATA_LAYOUT: usize = 3;

    /// Sampler + Sampled Images
    pub const SAMPLERS_AND_TEXTURES_LAYOUT: usize = 4;

    /// Storage buffer used for material colour data.
    pub const MATERIAL_COLOURS_LAYOUT: usize = 5;

    /// Storage buffer used for materials.
    pub const MATERIALS_LAYOUT: usize = 6;

    /// Storage buffer used for other textures besides image and constant colour.
    pub const OTHER_TEXTURES_LAYOUT: usize = 7;

    /// Uniform buffer for sky.
    pub const SKY_LAYOUT: usize = 8;

    /// Create a new raytracing pipeline.
    pub fn new(context: Arc<VulkanContext>) -> Result<Self> {
        let context = context.clone();

        // The order should match the `*_LAYOUT` constants.
        let set_layouts = vec![
            create_tlas_layout(context.clone())?,
            create_camera_layout(context.clone())?,
            create_render_image_layout(context.clone())?,
            create_mesh_data_layout(context.clone())?,
            create_sampler_and_image_textures_layout(context.clone())?,
            create_constant_colour_textures_layout(context.clone())?,
            create_materials_layout(context.clone())?,
            create_other_textures_layout(context.clone())?,
            create_sky_layout(context.clone())?,
        ];

        let push_constant_ranges = [
            vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)
                .offset(0)
                .size(size_of::<RayGenPushConstants>() as _),
            vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
                .offset(size_of::<RayGenPushConstants>() as _)
                .size(size_of::<ClosestHitPushConstants>() as _),
        ];

        let layouts: Vec<_> = set_layouts.iter().map(|layout| layout.get()).collect();

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(&push_constant_ranges);

        let pipeline_layout = unsafe {
            context
                .device
                .create_pipeline_layout(&pipeline_layout_create_info, None)?
        };

        let shader_modules = ShaderModules::load(context.clone())?;

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::RAYGEN_KHR)
                .module(shader_modules.ray_gen)
                .name(ENTRY_POINT),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::MISS_KHR)
                .module(shader_modules.ray_miss)
                .name(ENTRY_POINT),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
                .module(shader_modules.closest_hit)
                .name(ENTRY_POINT),
        ];

        let shader_groups = [
            // ray_gen
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(0)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
            // ray_miss
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(1)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
            // closest_hit
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                .closest_hit_shader(2)
                .general_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
        ];

        let rt_loader = khr::ray_tracing_pipeline::Device::new(&context.instance, &context.device);

        let pipeline_create_info = vk::RayTracingPipelineCreateInfoKHR::default()
            .stages(&shader_stages)
            .groups(&shader_groups)
            .max_pipeline_ray_recursion_depth(context.rt_pipeline_max_recursion_depth)
            .layout(pipeline_layout);

        let pipeline = unsafe {
            rt_loader
                .create_ray_tracing_pipelines(
                    vk::DeferredOperationKHR::null(),
                    vk::PipelineCache::null(),
                    &[pipeline_create_info],
                    None,
                )
                .map_err(|(_p, e)| anyhow!("Failed to create raytracing pipeline. {e:?}"))?
        }[0];

        let mut rt_pipeline_properties =
            vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();

        {
            let mut physical_device_properties2 =
                vk::PhysicalDeviceProperties2::default().push_next(&mut rt_pipeline_properties);

            unsafe {
                context.instance.get_physical_device_properties2(
                    context.physical_device,
                    &mut physical_device_properties2,
                );
            }
        }

        let handle_size_aligned = aligned_size(
            rt_pipeline_properties.shader_group_handle_size,
            rt_pipeline_properties.shader_group_base_alignment,
        );

        let incoming_table_data = unsafe {
            rt_loader.get_ray_tracing_shader_group_handles(
                pipeline,
                0,
                shader_groups.len() as u32,
                shader_groups.len() * rt_pipeline_properties.shader_group_handle_size as usize,
            )
        }
        .unwrap();

        let table_size = shader_groups.len() * handle_size_aligned as usize;
        let mut table_data = vec![0u8; table_size];

        for i in 0..shader_groups.len() {
            table_data[i * handle_size_aligned as usize
                ..i * handle_size_aligned as usize
                    + rt_pipeline_properties.shader_group_handle_size as usize]
                .copy_from_slice(
                    &incoming_table_data[i * rt_pipeline_properties.shader_group_handle_size
                        as usize
                        ..i * rt_pipeline_properties.shader_group_handle_size as usize
                            + rt_pipeline_properties.shader_group_handle_size as usize],
                );
        }

        let mut sbt_buffer = Buffer::new(
            context.clone(),
            table_size as u64,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;
        sbt_buffer.store(&table_data)?;

        // |[ ray gen shader ]|[ ray miss shader  ]|[ closest hit shader ]|
        // |                  |                    |                      |
        // | 0                | 1                  | 2                    | 3
        let sbt_address = sbt_buffer.get_buffer_device_address();

        let sbt_ray_gen_region = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(sbt_address)
            .size(handle_size_aligned)
            .stride(handle_size_aligned);

        let sbt_ray_miss_region = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(sbt_address + handle_size_aligned)
            .size(handle_size_aligned)
            .stride(handle_size_aligned);

        let sbt_closest_hit_region = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(sbt_address + 2 * handle_size_aligned)
            .size(handle_size_aligned)
            .stride(handle_size_aligned);

        let sbt_call_region = vk::StridedDeviceAddressRegionKHR::default();

        debug!("ray-gen SBT: {sbt_ray_gen_region:?}");
        debug!("ray-miss SBT: {sbt_ray_miss_region:?}");
        debug!("closest-hit SBT: {sbt_closest_hit_region:?}");

        Ok(Self {
            context,
            pipeline_layout,
            pipeline,
            set_layouts,
            rt_loader,
            sbt_ray_gen_region,
            sbt_ray_miss_region,
            sbt_closest_hit_region,
            sbt_call_region,
            _sbt_buffer: sbt_buffer,
        })
    }

    pub fn record_commands(
        &self,
        command_buffer: &CommandBuffer,
        descriptor_sets: &[vk::DescriptorSet],
        push_constants: &UnifiedPushConstants,
    ) {
        command_buffer.bind_pipeline(vk::PipelineBindPoint::RAY_TRACING_KHR, self.pipeline);

        command_buffer.bind_descriptor_sets(
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            self.pipeline_layout,
            descriptor_sets,
        );

        command_buffer.push_constants(
            self.pipeline_layout,
            vk::ShaderStageFlags::RAYGEN_KHR,
            push_constants.ray_gen_pc.to_raw_bytes(),
            0,
        );
        command_buffer.push_constants(
            self.pipeline_layout,
            vk::ShaderStageFlags::CLOSEST_HIT_KHR,
            push_constants.closest_hit_pc.to_raw_bytes(),
            std::mem::size_of::<RayGenPushConstants>() as _,
        );

        unsafe {
            self.rt_loader.cmd_trace_rays(
                command_buffer.get(),
                &self.sbt_ray_gen_region,
                &self.sbt_ray_miss_region,
                &self.sbt_closest_hit_region,
                &self.sbt_call_region,
                push_constants.ray_gen_pc.resolution[0],
                push_constants.ray_gen_pc.resolution[1],
                1,
            );
        }
    }
}

impl Drop for RtPipeline {
    fn drop(&mut self) {
        debug!("RtPipeline::drop()");
        unsafe {
            self.context.device.device_wait_idle().unwrap();

            self.context.device.destroy_pipeline(self.pipeline, None);

            self.context
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}

/// Create a pipeline layout for top level acceleration structure.
fn create_tlas_layout(context: Arc<VulkanContext>) -> Result<DescriptorSetLayout> {
    DescriptorSetLayout::new(
        context,
        &[vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR)],
        &[],
    )
}

/// Create a pipeline layout for uniform buffer containing camera matrices.
fn create_camera_layout(context: Arc<VulkanContext>) -> Result<DescriptorSetLayout> {
    DescriptorSetLayout::new(
        context,
        &[vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)],
        &[],
    )
}

/// Create a pipeline layout for the render image storage buffer.
fn create_render_image_layout(context: Arc<VulkanContext>) -> Result<DescriptorSetLayout> {
    DescriptorSetLayout::new(
        context,
        &[vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)],
        &[],
    )
}

/// Create a pipeline layout for mesh data references storage buffer.
fn create_mesh_data_layout(context: Arc<VulkanContext>) -> Result<DescriptorSetLayout> {
    // 0 - Vertex buffer.
    // 1 - Index buffer.
    // 2 - Meshes.
    let bindings: Vec<_> = (0..3)
        .map(|i| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
        })
        .collect();
    DescriptorSetLayout::new(context, &bindings, &[])
}

/// Create a pipeline layout for sampler and image textures.
fn create_sampler_and_image_textures_layout(
    context: Arc<VulkanContext>,
) -> Result<DescriptorSetLayout> {
    DescriptorSetLayout::new(
        context,
        &[
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR),
        ],
        &[
            vk::DescriptorBindingFlags::empty(), // for sampler
            vk::DescriptorBindingFlags::PARTIALLY_BOUND
                | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT, // texture images
        ],
    )
}

/// Create a pipeline layout for constant colour textures (this is just unique colour values).
fn create_constant_colour_textures_layout(
    context: Arc<VulkanContext>,
) -> Result<DescriptorSetLayout> {
    DescriptorSetLayout::new(
        context,
        &[vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR)],
        &[],
    )
}

/// Create a pipeline layout for material references storage buffer.
fn create_materials_layout(context: Arc<VulkanContext>) -> Result<DescriptorSetLayout> {
    // 0 - Lambertian materials.
    // 1 - Metal materials.
    // 2 - Dielectric materials.
    // 3 - Diffuse light materials.
    let bindings: Vec<_> = (0..4)
        .map(|i| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
        })
        .collect();

    DescriptorSetLayout::new(context, &bindings, &[])
}

/// Create a pipeline layout for storage buffer used for other textures besides image and constant colour.
fn create_other_textures_layout(context: Arc<VulkanContext>) -> Result<DescriptorSetLayout> {
    // 0 - Checker textures.
    // 1 - Noise textures.
    let bindings: Vec<_> = (0..2)
        .map(|i| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
        })
        .collect();

    DescriptorSetLayout::new(context, &bindings, &[])
}

/// Create a pipeline layout for uniform buffer containing sky.
fn create_sky_layout(context: Arc<VulkanContext>) -> Result<DescriptorSetLayout> {
    DescriptorSetLayout::new(
        context,
        &[vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)],
        &[],
    )
}

fn aligned_size(value: u32, alignment: u32) -> u64 {
    ((value + alignment - 1) & !(alignment - 1)) as u64
}
