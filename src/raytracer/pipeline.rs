use anyhow::Result;
use std::sync::Arc;
use vulkano::{
    descriptor_set::layout::{
        DescriptorBindingFlags, DescriptorSetLayout, DescriptorSetLayoutBinding,
        DescriptorSetLayoutCreateInfo, DescriptorType,
    },
    device::Device,
    pipeline::{
        PipelineLayout, PipelineShaderStageCreateInfo,
        layout::PipelineLayoutCreateInfo,
        ray_tracing::{
            RayTracingPipeline, RayTracingPipelineCreateInfo, RayTracingShaderGroupCreateInfo,
        },
    },
    shader::ShaderStages,
};

pub struct RtPipeline {
    pipeline: Arc<RayTracingPipeline>,
    pipeline_layout: Arc<PipelineLayout>,
}

impl RtPipeline {
    pub const TLAS_LAYOUT: usize = 0;
    pub const UNIFORM_BUFFER_LAYOUT: usize = 1;
    pub const RENDER_IMAGE_LAYOUT: usize = 2;
    pub const MESH_DATA_LAYOUT: usize = 3;
    pub const SAMPLERS_AND_TEXTURES_LAYOUT: usize = 4;

    pub fn get(&self) -> Arc<RayTracingPipeline> {
        self.pipeline.clone()
    }

    pub fn get_layout(&self) -> Arc<PipelineLayout> {
        self.pipeline_layout.clone()
    }

    pub fn new(
        device: Arc<Device>,
        stages: &[PipelineShaderStageCreateInfo],
        groups: &[RayTracingShaderGroupCreateInfo],
        texture_count: u32,
    ) -> Result<Self> {
        let pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![
                    // The order should match the `*_LAYOUT` constants.
                    create_tlas_layout(device.clone()),
                    create_uniform_buffer_layout(device.clone()),
                    create_render_image_layout(device.clone()),
                    create_mesh_data_layout(device.clone()),
                    create_sample_and_textures_layout(device.clone(), texture_count),
                ],
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
