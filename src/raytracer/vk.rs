use std::sync::Arc;

use anyhow::{Result, anyhow};
use vulkano::{
    DeviceSize,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryCommandBufferAbstract,
        allocator::CommandBufferAllocator,
    },
    descriptor_set::allocator::DescriptorSetAllocator,
    device::{Device, Queue},
    memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter},
    sync::GpuFuture,
};

/// Our own vulkano context. Wraps some common resources we will want to use.
pub struct Vk {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub memory_allocator: Arc<dyn MemoryAllocator>,
    pub command_buffer_allocator: Arc<dyn CommandBufferAllocator>,
    pub descriptor_set_allocator: Arc<dyn DescriptorSetAllocator>,
}

/// This will create buffers that can be accessed only by the GPU. One specific use case is to
/// access them via device addresses in shaders.
pub fn create_device_local_buffer<T, I>(
    vk: Arc<Vk>,
    usage: BufferUsage,
    data: I,
) -> Result<Subbuffer<[T]>>
where
    T: BufferContents,
    I: IntoIterator<Item = T>,
    I::IntoIter: ExactSizeIterator,
{
    let iter = data.into_iter();
    let size = iter.len() as DeviceSize;

    if size == 0 {
        return Err(anyhow!("Cannot create device local buffer with empty data"));
    }

    let temporary_accessible_buffer = Buffer::from_iter(
        vk.memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        iter,
    )?;

    let device_local_buffer = Buffer::new_slice::<T>(
        vk.memory_allocator.clone(),
        BufferCreateInfo {
            usage: usage | BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        size,
    )?;

    let mut builder = AutoCommandBufferBuilder::primary(
        vk.command_buffer_allocator.clone(),
        vk.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )?;

    builder.copy_buffer(CopyBufferInfo::buffers(
        temporary_accessible_buffer,
        device_local_buffer.clone(),
    ))?;

    builder
        .build()?
        .execute(vk.queue.clone())?
        .then_signal_fence_and_flush()?
        .wait(None /* timeout */)?;

    Ok(device_local_buffer)
}
