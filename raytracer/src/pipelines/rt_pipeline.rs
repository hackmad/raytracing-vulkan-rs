use std::sync::Arc;

use anyhow::Result;
use shaders::{any_hit, closest_hit, intersection, ray_gen};
use vulkano::{
    descriptor_set::layout::{
        DescriptorBindingFlags, DescriptorSetLayout, DescriptorSetLayoutBinding,
        DescriptorSetLayoutCreateInfo, DescriptorType,
    },
    device::Device,
    pipeline::{
        PipelineLayout, PipelineShaderStageCreateInfo,
        layout::{PipelineLayoutCreateInfo, PushConstantRange},
        ray_tracing::{
            RayTracingPipeline, RayTracingPipelineCreateInfo, RayTracingShaderGroupCreateInfo,
        },
    },
    shader::ShaderStages,
};

/// The raytracing pipeline.
pub struct RtPipeline {
    /// The pipeline.
    pipeline: Arc<RayTracingPipeline>,

    /// The pipeline layout.
    pipeline_layout: Arc<PipelineLayout>,
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

    /// Storage buffer for light source alias table.
    pub const LIGHT_SOURCE_ALIAS_TABLE: usize = 9;

    /// Storage buffer used for volume data.
    pub const VOLUME_DATA_LAYOUT: usize = 10;

    /// Returns the pipeline.
    pub fn get(&self) -> Arc<RayTracingPipeline> {
        self.pipeline.clone()
    }

    /// Returns the pipeline layout.
    pub fn get_layout(&self) -> Arc<PipelineLayout> {
        self.pipeline_layout.clone()
    }

    /// Create a new raytracing pipeline.
    pub fn new(
        device: Arc<Device>,
        stages: &[PipelineShaderStageCreateInfo],
        groups: &[RayTracingShaderGroupCreateInfo],
        image_texture_count: u32,
    ) -> Result<Self> {
        let ray_gen_pc = PushConstantRange {
            stages: ShaderStages::RAYGEN,
            offset: 0,
            size: size_of::<ray_gen::PushConstants>() as _,
        };

        let closest_hit_pc = PushConstantRange {
            stages: ShaderStages::CLOSEST_HIT,
            offset: ray_gen_pc.size,
            size: size_of::<closest_hit::PushConstants>() as _,
        };

        let intersection_pc = PushConstantRange {
            stages: ShaderStages::INTERSECTION,
            offset: ray_gen_pc.size + closest_hit_pc.size,
            size: size_of::<intersection::PushConstants>() as _,
        };

        let any_hit_pc = PushConstantRange {
            stages: ShaderStages::ANY_HIT,
            offset: ray_gen_pc.size + closest_hit_pc.size + intersection_pc.size,
            size: size_of::<any_hit::PushConstants>() as _,
        };

        let push_constant_ranges = vec![ray_gen_pc, closest_hit_pc, intersection_pc, any_hit_pc];

        let pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![
                    // The order should match the `*_LAYOUT` constants.
                    create_tlas_layout(device.clone()),
                    create_camera_layout(device.clone()),
                    create_render_image_layout(device.clone()),
                    create_mesh_data_layout(device.clone()),
                    create_sampler_and_image_textures_layout(device.clone(), image_texture_count),
                    create_constant_colour_textures_layout(device.clone()),
                    create_materials_layout(device.clone()),
                    create_other_textures_layout(device.clone()),
                    create_sky_layout(device.clone()),
                    create_light_source_alias_table_layout(device.clone()),
                    create_volume_data_layout(device.clone()),
                ],
                push_constant_ranges,
                ..Default::default()
            },
        )?;

        let pipeline = RayTracingPipeline::new(
            device.clone(),
            None,
            RayTracingPipelineCreateInfo {
                stages: stages.into(),
                groups: groups.into(),
                max_pipeline_ray_recursion_depth: 1,
                ..RayTracingPipelineCreateInfo::layout(pipeline_layout.clone())
            },
        )?;

        Ok(Self {
            pipeline,
            pipeline_layout,
        })
    }
}

/// Create a pipeline layout for top level acceleration structure.
fn create_tlas_layout(device: Arc<Device>) -> Arc<DescriptorSetLayout> {
    DescriptorSetLayout::new(
        device,
        DescriptorSetLayoutCreateInfo {
            #[rustfmt::skip]
            bindings: [(0, as_binding(ShaderStages::RAYGEN))]
                .into_iter()
                .collect(),
            ..Default::default()
        },
    )
    .unwrap()
}

/// Create a pipeline layout for uniform buffer containing camera matrices.
fn create_camera_layout(device: Arc<Device>) -> Arc<DescriptorSetLayout> {
    DescriptorSetLayout::new(
        device,
        DescriptorSetLayoutCreateInfo {
            bindings: [(0, uniform_buffer_binding(ShaderStages::RAYGEN))]
                .into_iter()
                .collect(),
            ..Default::default()
        },
    )
    .unwrap()
}

/// Create a pipeline layout for the render image storage buffer.
fn create_render_image_layout(device: Arc<Device>) -> Arc<DescriptorSetLayout> {
    DescriptorSetLayout::new(
        device.clone(),
        DescriptorSetLayoutCreateInfo {
            bindings: [(0, storage_image_binding(ShaderStages::RAYGEN))]
                .into_iter()
                .collect(),
            ..Default::default()
        },
    )
    .unwrap()
}

/// Create a pipeline layout for mesh data references storage buffer.
fn create_mesh_data_layout(device: Arc<Device>) -> Arc<DescriptorSetLayout> {
    DescriptorSetLayout::new(
        device.clone(),
        DescriptorSetLayoutCreateInfo {
            bindings: [
                (0, storage_buffer_binding(ShaderStages::CLOSEST_HIT)), // Vertex buffer.
                (1, storage_buffer_binding(ShaderStages::CLOSEST_HIT)), // Index buffer.
                (2, storage_buffer_binding(ShaderStages::CLOSEST_HIT)), // Meshes.
            ]
            .into_iter()
            .collect(),
            ..Default::default()
        },
    )
    .unwrap()
}

/// Create a pipeline layout for sampler and image textures.
fn create_sampler_and_image_textures_layout(
    device: Arc<Device>,
    image_texture_count: u32,
) -> Arc<DescriptorSetLayout> {
    let shader_stages = ShaderStages::CLOSEST_HIT | ShaderStages::ANY_HIT;

    DescriptorSetLayout::new(
        device.clone(),
        DescriptorSetLayoutCreateInfo {
            #[rustfmt::skip]
            bindings: [
                (0, sampler_binding(shader_stages)),
                (1, variable_sampled_image_binding(shader_stages, image_texture_count)),
            ]
            .into_iter()
            .collect(),
            ..Default::default()
        },
    )
    .unwrap()
}

/// Create a pipeline layout for constant colour textures (this is just unique colour values).
fn create_constant_colour_textures_layout(device: Arc<Device>) -> Arc<DescriptorSetLayout> {
    let shader_stages = ShaderStages::CLOSEST_HIT | ShaderStages::ANY_HIT;

    DescriptorSetLayout::new(
        device.clone(),
        DescriptorSetLayoutCreateInfo {
            bindings: [(0, storage_buffer_binding(shader_stages))]
                .into_iter()
                .collect(),
            ..Default::default()
        },
    )
    .unwrap()
}

/// Create a pipeline layout for material references storage buffer.
fn create_materials_layout(device: Arc<Device>) -> Arc<DescriptorSetLayout> {
    let shader_stages = ShaderStages::CLOSEST_HIT | ShaderStages::ANY_HIT;

    DescriptorSetLayout::new(
        device.clone(),
        DescriptorSetLayoutCreateInfo {
            bindings: [
                (0, storage_buffer_binding(shader_stages)), // Lambertian materials.
                (1, storage_buffer_binding(shader_stages)), // Metal materials.
                (2, storage_buffer_binding(shader_stages)), // Dielectric materials.
                (3, storage_buffer_binding(shader_stages)), // Diffuse light materials.
                (4, storage_buffer_binding(shader_stages)), // Isotropic materials.
            ]
            .into_iter()
            .collect(),
            ..Default::default()
        },
    )
    .unwrap()
}

/// Create a pipeline layout for storage buffer used for other textures besides image and constant colour.
fn create_other_textures_layout(device: Arc<Device>) -> Arc<DescriptorSetLayout> {
    let shader_stages = ShaderStages::CLOSEST_HIT | ShaderStages::ANY_HIT;

    DescriptorSetLayout::new(
        device.clone(),
        DescriptorSetLayoutCreateInfo {
            bindings: [
                (0, storage_buffer_binding(shader_stages)), // Checker textures.
                (1, storage_buffer_binding(shader_stages)), // Noise textures.
            ]
            .into_iter()
            .collect(),
            ..Default::default()
        },
    )
    .unwrap()
}

/// Create a pipeline layout for uniform buffer containing sky.
fn create_sky_layout(device: Arc<Device>) -> Arc<DescriptorSetLayout> {
    DescriptorSetLayout::new(
        device,
        DescriptorSetLayoutCreateInfo {
            bindings: [(0, uniform_buffer_binding(ShaderStages::RAYGEN))]
                .into_iter()
                .collect(),
            ..Default::default()
        },
    )
    .unwrap()
}

/// Create a pipeline layout for light source alias table storage buffer.
fn create_light_source_alias_table_layout(device: Arc<Device>) -> Arc<DescriptorSetLayout> {
    DescriptorSetLayout::new(
        device.clone(),
        DescriptorSetLayoutCreateInfo {
            bindings: [(0, storage_buffer_binding(ShaderStages::CLOSEST_HIT))]
                .into_iter()
                .collect(),
            ..Default::default()
        },
    )
    .unwrap()
}

/// Create a pipeline layout for volume data references storage buffer.
fn create_volume_data_layout(device: Arc<Device>) -> Arc<DescriptorSetLayout> {
    let binding = ShaderStages::INTERSECTION | ShaderStages::ANY_HIT;

    DescriptorSetLayout::new(
        device.clone(),
        DescriptorSetLayoutCreateInfo {
            bindings: [
                (0, storage_buffer_binding(binding)), // Volume metadata
                (1, storage_buffer_binding(binding)), // Sphere Volume metadata
                (2, storage_buffer_binding(binding)), // Constant density media
            ]
            .into_iter()
            .collect(),
            ..Default::default()
        },
    )
    .unwrap()
}

fn as_binding(stages: ShaderStages) -> DescriptorSetLayoutBinding {
    DescriptorSetLayoutBinding {
        stages,
        ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::AccelerationStructure)
    }
}

fn uniform_buffer_binding(stages: ShaderStages) -> DescriptorSetLayoutBinding {
    DescriptorSetLayoutBinding {
        stages,
        ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBuffer)
    }
}

fn storage_image_binding(stages: ShaderStages) -> DescriptorSetLayoutBinding {
    DescriptorSetLayoutBinding {
        stages,
        ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageImage)
    }
}

fn sampler_binding(stages: ShaderStages) -> DescriptorSetLayoutBinding {
    DescriptorSetLayoutBinding {
        stages,
        ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::Sampler)
    }
}

fn variable_sampled_image_binding(stages: ShaderStages, count: u32) -> DescriptorSetLayoutBinding {
    DescriptorSetLayoutBinding {
        stages,
        binding_flags: DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT,
        descriptor_count: count,
        ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::SampledImage)
    }
}

fn storage_buffer_binding(stages: ShaderStages) -> DescriptorSetLayoutBinding {
    DescriptorSetLayoutBinding {
        stages,
        ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer)
    }
}
