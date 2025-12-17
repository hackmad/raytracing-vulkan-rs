use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use log::debug;
use vulkano::{
    DeviceSize,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryCommandBufferAbstract,
        allocator::CommandBufferAllocator,
    },
    descriptor_set::allocator::DescriptorSetAllocator,
    device::{Device, Queue},
    memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryAllocator, MemoryTypeFilter},
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
    let size = iter.len();
    let size_bytes = (size * size_of::<T>()) as DeviceSize;

    if size == 0 {
        return Err(anyhow!("Cannot create device local buffer with empty data"));
    }

    // Create a memory layout so the scratch buffer address is aligned correctly for the storage buffer.
    let device_properties = vk.device.physical_device().properties();
    let min_scratch_offset = device_properties.min_storage_buffer_offset_alignment.into();
    let buffer_layout = DeviceLayout::from_size_alignment(size_bytes, min_scratch_offset)
        .context("Unable to create buffer device layout")?;

    debug!("Storage buffer min_storage_buffer_offset_alignment: {min_scratch_offset}");
    debug!("Storage buffer size: {size} ({size_bytes} bytes)");
    debug!("Storage buffer layout: {:?}", buffer_layout);

    let scratch_buffer = Subbuffer::new(Buffer::new(
        vk.memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC | BufferUsage::SHADER_DEVICE_ADDRESS,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        buffer_layout,
    )?)
    .reinterpret::<[T]>();

    {
        let mut write_guard = scratch_buffer.write()?;
        for (o, i) in write_guard.iter_mut().zip(iter) {
            *o = i;
        }
    }

    let scratch_buffer_address: u64 = scratch_buffer.device_address()?.into();
    debug!(
        "Scratch buffer device addr: {scratch_buffer_address} is {}",
        if scratch_buffer_address.is_multiple_of(min_scratch_offset) {
            "aligned"
        } else {
            "NOT ALIGNED"
        }
    );

    let device_local_buffer = Subbuffer::new(Buffer::new(
        vk.memory_allocator.clone(),
        BufferCreateInfo {
            usage: usage | BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        buffer_layout,
    )?)
    .reinterpret::<[T]>();

    let device_local_buffer_address: u64 = device_local_buffer.device_address()?.into();
    debug!(
        "Device local buffer device addr: {device_local_buffer_address} is {}",
        if device_local_buffer_address.is_multiple_of(min_scratch_offset) {
            "aligned"
        } else {
            "NOT ALIGNED"
        }
    );

    let mut builder = AutoCommandBufferBuilder::primary(
        vk.command_buffer_allocator.clone(),
        vk.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )?;

    builder.copy_buffer(CopyBufferInfo::buffers(
        scratch_buffer,
        device_local_buffer.clone(),
    ))?;

    builder
        .build()?
        .execute(vk.queue.clone())?
        .then_signal_fence_and_flush()?
        .wait(None /* timeout */)?;

    Ok(device_local_buffer)
}
