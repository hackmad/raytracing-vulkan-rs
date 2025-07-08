use std::sync::Arc;

use anyhow::Result;
use ash::vk;
use image::RgbaImage;

use crate::{Buffer, CommandBuffer, NO_FENCE, VulkanContext, get_memory_type_index};

pub struct Image {
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub width: u32,
    pub height: u32,

    context: Arc<VulkanContext>,
    image_memory: Option<vk::DeviceMemory>,
    is_external_alloc: bool,
}

impl Image {
    /// Wraps a vk::Image and vk::ImageView. It does not require any memory allocations.
    pub fn new(
        context: Arc<VulkanContext>,
        image: vk::Image,
        image_view: vk::ImageView,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            context,
            image,
            image_view,
            width,
            height,
            image_memory: None,
            is_external_alloc: true,
        }
    }

    pub fn new_rgba_image(context: Arc<VulkanContext>, rgba_image: &RgbaImage) -> Result<Self> {
        let (width, height) = rgba_image.dimensions();
        let format = vk::Format::R8G8B8A8_SRGB;
        let buffer_size = (width * height * 4) as vk::DeviceSize;

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = unsafe { context.device.create_image(&image_info, None)? };

        let mem_requirements = unsafe { context.device.get_image_memory_requirements(image) };

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(get_memory_type_index(
                context.device_memory_properties,
                mem_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ));

        let image_memory = unsafe { context.device.allocate_memory(&alloc_info, None)? };

        unsafe {
            context.device.bind_image_memory(image, image_memory, 0)?;
        }

        let mut staging_buffer = Buffer::new(
            context.clone(),
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        staging_buffer.store(rgba_image.as_raw())?;

        transition_image_layout(
            context.clone(),
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        )?;

        copy_buffer_to_image(context.clone(), staging_buffer, image, width, height)?;

        transition_image_layout(
            context.clone(),
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        )?;

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        let image_view = unsafe { context.device.create_image_view(&view_info, None)? };

        Ok(Self {
            context,
            image,
            image_view,
            width,
            height,
            image_memory: Some(image_memory),
            is_external_alloc: false,
        })
    }

    pub fn new_render_image(context: Arc<VulkanContext>, width: u32, height: u32) -> Result<Self> {
        let format = vk::Format::R8G8B8A8_UNORM;

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::STORAGE,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = unsafe { context.device.create_image(&image_info, None)? };

        let mem_requirements = unsafe { context.device.get_image_memory_requirements(image) };

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(get_memory_type_index(
                context.device_memory_properties,
                mem_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            ));

        let image_memory = unsafe { context.device.allocate_memory(&alloc_info, None)? };

        unsafe {
            context.device.bind_image_memory(image, image_memory, 0)?;
        }

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        let image_view = unsafe { context.device.create_image_view(&view_info, None)? };

        Ok(Self {
            context,
            image,
            image_view,
            width,
            height,
            image_memory: Some(image_memory),
            is_external_alloc: false,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn transition_layout(
        &self,
        command_buffer: &CommandBuffer,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        src_stage_mask: vk::PipelineStageFlags,
        dst_stage_mask: vk::PipelineStageFlags,
        src_access_mask: vk::AccessFlags,
        dst_access_mask: vk::AccessFlags,
    ) {
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_access_mask(src_access_mask)
            .dst_access_mask(dst_access_mask);

        command_buffer.pipeline_image_memory_barrier(
            barrier,
            src_stage_mask,
            dst_stage_mask,
            vk::DependencyFlags::empty(),
        );
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        if !self.is_external_alloc {
            unsafe {
                self.context
                    .device
                    .destroy_image_view(self.image_view, None);

                self.context.device.destroy_image(self.image, None);

                if let Some(image_memory) = self.image_memory {
                    self.context.device.free_memory(image_memory, None);
                }
            }
        }
    }
}

fn transition_image_layout(
    context: Arc<VulkanContext>,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) -> Result<()> {
    let command_buffer = CommandBuffer::new(context.clone())?;
    command_buffer.begin_one_time_submit()?;

    let (src_access_mask, dst_access_mask, src_stage, dst_stage) = match (old_layout, new_layout) {
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
        ),
        (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        ),
        _ => panic!("Unsupported layout transition!"),
    };

    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        )
        .src_access_mask(src_access_mask)
        .dst_access_mask(dst_access_mask);

    command_buffer.pipeline_image_memory_barrier(
        barrier,
        src_stage,
        dst_stage,
        vk::DependencyFlags::empty(),
    );

    command_buffer.end()?;

    command_buffer.submit(None, &NO_FENCE)?;

    Ok(())
}

fn copy_buffer_to_image(
    context: Arc<VulkanContext>,
    buffer: Buffer,
    image: vk::Image,
    width: u32,
    height: u32,
) -> Result<()> {
    let command_buffer = CommandBuffer::new(context.clone())?;
    command_buffer.begin_one_time_submit()?;

    let region = vk::BufferImageCopy::default()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1),
        )
        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
        .image_extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        });

    command_buffer.copy_buffer_to_image(
        &buffer,
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &[region],
    );

    command_buffer.end()?;

    command_buffer.submit(None, &NO_FENCE)?;

    Ok(())
}
