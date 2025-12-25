use std::sync::Arc;

use anyhow::{Context, Result};
use foldhash::{HashSet, fast::RandomState};
use vulkano::{
    descriptor_set::layout::{
        DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
        DescriptorType,
    },
    device::Device,
    format::Format,
    pipeline::{
        DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::VertexInputState,
            viewport::{Viewport, ViewportState},
        },
        layout::{PipelineDescriptorSetLayoutCreateInfo, PipelineLayoutCreateInfo},
    },
    render_pass::{RenderPass, Subpass},
    shader::ShaderStages,
};

/// The graphics pipeline used for copying rendered image from RayTracingPipeline which is in
/// linear colour space to the Swapchain which is using sRGB colour space.
pub struct GfxPipeline {
    /// The pipeline.
    pipeline: Arc<GraphicsPipeline>,

    /// The pipeline layout.
    pipeline_layout: Arc<PipelineLayout>,

    /// Render pass.
    render_pass: Arc<RenderPass>,
}

impl GfxPipeline {
    // These make it easier to set the descriptor set layout.

    /// Render image.
    pub const RENDER_IMAGE_LAYOUT: usize = 0;

    /// Returns the pipeline.
    pub fn get(&self) -> Arc<GraphicsPipeline> {
        self.pipeline.clone()
    }

    /// Returns the pipeline layout.
    pub fn get_layout(&self) -> Arc<PipelineLayout> {
        self.pipeline_layout.clone()
    }

    /// Returns the render pass.
    pub fn get_render_pass(&self) -> Arc<RenderPass> {
        self.render_pass.clone()
    }

    /// Create a new graphics pipeline.
    pub fn new(
        device: Arc<Device>,
        stages: &[PipelineShaderStageCreateInfo],
        window_size: &[f32; 2],
        swapchain_format: Format,
    ) -> Result<Self> {
        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: *window_size,
            depth_range: 0.0..=1.0,
        };

        let descriptor_set_ci = PipelineDescriptorSetLayoutCreateInfo::from_stages(stages);
        let layout_ci = descriptor_set_ci.into_pipeline_layout_create_info(device.clone())?;
        let layout = PipelineLayout::new(device.clone(), layout_ci)?;

        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    format: swapchain_format,
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            },
        )?;
        let subpass = Subpass::from(render_pass.clone(), 0)
            .with_context(|| "Failed to create graphics pipeline subpass from render pass")?;

        let pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![
                    // The order should match the `*_LAYOUT` constants.
                    create_render_image_layout(device.clone()),
                ],
                push_constant_ranges: vec![],
                ..Default::default()
            },
        )?;

        let mut dynamic_state = HashSet::with_hasher(RandomState::default());
        dynamic_state.insert(DynamicState::Viewport);

        let pipeline = GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into(),
                vertex_input_state: Some(VertexInputState::new()), // explicitly "no vertex input"
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState {
                    viewports: [viewport].into_iter().collect(),
                    ..Default::default()
                }),
                dynamic_state,
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )?;

        Ok(Self {
            pipeline,
            pipeline_layout,
            render_pass,
        })
    }
}

/// Create a pipeline layout for the combined image + sampler for the render image.
fn create_render_image_layout(device: Arc<Device>) -> Arc<DescriptorSetLayout> {
    DescriptorSetLayout::new(
        device.clone(),
        DescriptorSetLayoutCreateInfo {
            bindings: [(
                0,
                DescriptorSetLayoutBinding {
                    descriptor_count: 1,
                    stages: ShaderStages::FRAGMENT,
                    ..DescriptorSetLayoutBinding::descriptor_type(
                        DescriptorType::CombinedImageSampler,
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
