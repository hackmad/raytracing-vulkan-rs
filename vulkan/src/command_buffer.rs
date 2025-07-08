use std::sync::Arc;

use crate::{Buffer, Fence, VulkanContext};
use anyhow::Result;
use ash::vk::{self, Handle};
use log::debug;

pub struct CommandBuffer {
    context: Arc<VulkanContext>,
    command_buffer: vk::CommandBuffer,
    name: String,
}

impl CommandBuffer {
    pub fn new(context: Arc<VulkanContext>, name: &str) -> Result<Self> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(context.command_pool)
            .command_buffer_count(1);

        let command_buffer = unsafe { context.device.allocate_command_buffers(&alloc_info)? }[0];

        Ok(Self {
            context,
            command_buffer,
            name: name.to_string(),
        })
    }

    pub fn get(&self) -> vk::CommandBuffer {
        self.command_buffer
    }

    pub fn begin_one_time_submit(&self) -> Result<()> {
        debug!("Command buffer {}: begin", &self.name);

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.context
                .device
                .begin_command_buffer(self.command_buffer, &begin_info)?;
        }

        Ok(())
    }

    pub fn submit_and_wait(
        &self,
        submit_info: Option<vk::SubmitInfo>,
        fence: &Fence,
    ) -> Result<()> {
        debug!("Command buffer {}: submit_and_wait", &self.name);

        let graphics_queue = self.context.get_graphics_queue();
        let command_buffers = [self.command_buffer];

        unsafe {
            let submit_info = submit_info
                .unwrap_or_default()
                .command_buffers(&command_buffers);

            debug!("Command buffer {}: submitting", &self.name);

            let submit_result =
                self.context
                    .device
                    .queue_submit(graphics_queue, &[submit_info], fence.get());

            debug!(
                "Command buffer {}, queue_submit result: {submit_result:?}",
                &self.name
            );

            submit_result?;

            if !fence.get().is_null() {
                debug!(
                    "Command buffer {}: waiting for fence after submit",
                    &self.name
                );
                self.context
                    .device
                    .wait_for_fences(&[fence.get()], true, u64::MAX)?;
            } else {
                debug!(
                    "Command buffer {}: waiting for queue_wait_idle after submit",
                    &self.name
                );
                self.context.device.queue_wait_idle(graphics_queue)?;
            }
        }

        Ok(())
    }

    pub fn end(&self) -> Result<()> {
        debug!("Command buffer {}: end", &self.name);
        unsafe {
            self.context
                .device
                .end_command_buffer(self.command_buffer)?;
        }

        Ok(())
    }

    pub fn reset(&self) -> Result<()> {
        debug!("Command buffer {}: reset", &self.name);
        unsafe {
            self.context
                .device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;
        }
        Ok(())
    }

    pub fn pipeline_image_memory_barrier(
        &self,
        barrier: vk::ImageMemoryBarrier,
        src_stage: vk::PipelineStageFlags,
        dst_stage: vk::PipelineStageFlags,
        dependency_flags: vk::DependencyFlags,
    ) {
        unsafe {
            self.context.device.cmd_pipeline_barrier(
                self.command_buffer,
                src_stage,
                dst_stage,
                dependency_flags,
                &[],
                &[],
                &[barrier],
            );
        }
    }

    pub fn memory_barrier(
        &self,
        barrier: vk::MemoryBarrier,
        src_stage: vk::PipelineStageFlags,
        dst_stage: vk::PipelineStageFlags,
        dependency_flags: vk::DependencyFlags,
    ) {
        unsafe {
            self.context.device.cmd_pipeline_barrier(
                self.command_buffer,
                src_stage,
                dst_stage,
                dependency_flags,
                &[barrier],
                &[],
                &[],
            );
        }
    }

    pub fn copy_buffer_to_image(
        &self,
        buffer: &Buffer,
        image: vk::Image,
        dst_image_layout: vk::ImageLayout,
        regions: &[vk::BufferImageCopy],
    ) {
        unsafe {
            self.context.device.cmd_copy_buffer_to_image(
                self.command_buffer,
                buffer.buffer,
                image,
                dst_image_layout,
                regions,
            );
        }
    }

    pub fn copy_buffer(&self, src: &Buffer, dst: &Buffer, regions: &[vk::BufferCopy]) {
        unsafe {
            self.context.device.cmd_copy_buffer(
                self.command_buffer,
                src.buffer,
                dst.buffer,
                regions,
            );
        }
    }

    pub fn bind_pipeline(
        &self,
        pipeline_bind_point: vk::PipelineBindPoint,
        pipeline: vk::Pipeline,
    ) {
        unsafe {
            self.context.device.cmd_bind_pipeline(
                self.command_buffer,
                pipeline_bind_point,
                pipeline,
            );
        }
    }

    pub fn bind_descriptor_sets(
        &self,
        pipeline_bind_point: vk::PipelineBindPoint,
        pipeline_layout: vk::PipelineLayout,
        descriptor_sets: &[vk::DescriptorSet],
    ) {
        unsafe {
            self.context.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                pipeline_bind_point,
                pipeline_layout,
                0,
                descriptor_sets,
                &[],
            );
        }
    }

    pub fn push_constants(
        &self,
        pipeline_layout: vk::PipelineLayout,
        stage_flags: vk::ShaderStageFlags,
        push_constants: &[u8],
        offset: u32,
    ) {
        unsafe {
            self.context.device.cmd_push_constants(
                self.command_buffer,
                pipeline_layout,
                stage_flags,
                offset,
                push_constants,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn blit_image(
        &self,
        src_image: vk::Image,
        dst_image: vk::Image,
        src_layout: vk::ImageLayout,
        dst_layout: vk::ImageLayout,
        src_extent: vk::Extent3D,
        dst_extent: vk::Extent3D,
        filter: vk::Filter,
    ) {
        let blit = vk::ImageBlit::default()
            .src_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: src_extent.width as i32,
                    y: src_extent.height as i32,
                    z: src_extent.depth as i32,
                },
            ])
            .dst_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .dst_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: dst_extent.width as i32,
                    y: dst_extent.height as i32,
                    z: dst_extent.depth as i32,
                },
            ]);

        unsafe {
            self.context.device.cmd_blit_image(
                self.command_buffer,
                src_image,
                src_layout,
                dst_image,
                dst_layout,
                &[blit],
                filter,
            );
        }
    }
}

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        debug!("Command buffer {}: drop", &self.name);
        unsafe {
            self.context
                .device
                .free_command_buffers(self.context.command_pool, &[self.command_buffer]);
        }
    }
}
