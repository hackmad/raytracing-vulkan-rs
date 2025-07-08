use std::{ffi::c_void, sync::Arc};

use anyhow::Result;
use ash::{util::Align, vk};

use crate::{CommandBuffer, NO_FENCE, VulkanContext};

pub struct Buffer {
    pub buffer: vk::Buffer,

    context: Arc<VulkanContext>,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
}

impl Buffer {
    pub fn new(
        context: Arc<VulkanContext>,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_properties: vk::MemoryPropertyFlags,
    ) -> Result<Self> {
        unsafe {
            let buffer_info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let buffer = context.device.create_buffer(&buffer_info, None).unwrap();

            let memory_req = context.device.get_buffer_memory_requirements(buffer);

            let memory_index = get_memory_type_index(
                context.device_memory_properties,
                memory_req.memory_type_bits,
                memory_properties,
            );

            let mut memory_allocate_flags_info = vk::MemoryAllocateFlagsInfo::default()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);

            let mut allocate_info_builder = vk::MemoryAllocateInfo::default();

            if usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
                allocate_info_builder =
                    allocate_info_builder.push_next(&mut memory_allocate_flags_info);
            }

            let allocate_info = allocate_info_builder
                .allocation_size(memory_req.size)
                .memory_type_index(memory_index);

            let memory = context.device.allocate_memory(&allocate_info, None)?;

            context.device.bind_buffer_memory(buffer, memory, 0)?;

            Ok(Self {
                context,
                buffer,
                memory,
                size,
            })
        }
    }

    pub fn store<T: Copy>(&mut self, data: &[T]) -> Result<()> {
        unsafe {
            let size = std::mem::size_of_val(data) as u64;
            assert!(self.size >= size, "Data size is larger than buffer size.");

            let mapped_ptr = self.map(size)?;
            let mut mapped_slice = Align::new(mapped_ptr, std::mem::align_of::<T>() as u64, size);
            mapped_slice.copy_from_slice(data);
            self.unmap();

            Ok(())
        }
    }

    fn map(&mut self, size: vk::DeviceSize) -> Result<*mut c_void> {
        assert!(size > 0, "Buffer::map() called with size=0");

        unsafe {
            let data: *mut c_void = self.context.device.map_memory(
                self.memory,
                0,
                size,
                vk::MemoryMapFlags::empty(),
            )?;
            Ok(data)
        }
    }

    fn unmap(&mut self) {
        unsafe {
            self.context.device.unmap_memory(self.memory);
        }
    }

    pub fn get_buffer_device_address(&self) -> u64 {
        let buffer_device_address_info = vk::BufferDeviceAddressInfo::default().buffer(self.buffer);
        unsafe {
            self.context
                .device
                .get_buffer_device_address(&buffer_device_address_info)
        }
    }

    /// This will create buffers that can be accessed only by the GPU. One specific use case is to
    /// access them via device addresses in shaders.
    pub fn new_device_local_storage_buffer<T: Copy>(
        context: Arc<VulkanContext>,
        usage: vk::BufferUsageFlags,
        data: &[T],
    ) -> Result<Self> {
        let data_size = std::mem::size_of_val(data);

        let buffer_size = data_size.max(1) as vk::DeviceSize;
        let device_local_buffer = Buffer::new(
            context.clone(),
            buffer_size,
            usage | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        if data_size > 0 {
            let mut staging_buffer = Self::new(
                context.clone(),
                buffer_size,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            staging_buffer.store(data)?;

            let command_buffer = CommandBuffer::new(context.clone())?;
            command_buffer.begin_one_time_submit()?;

            let copy_regions = [vk::BufferCopy::default().size(buffer_size)];
            command_buffer.copy_buffer(&staging_buffer, &device_local_buffer, &copy_regions);

            command_buffer.end()?;

            command_buffer.submit(None, &NO_FENCE)?;
        }

        Ok(device_local_buffer)
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.context.device.destroy_buffer(self.buffer, None);
            self.context.device.free_memory(self.memory, None);
        }
    }
}

pub fn get_memory_type_index(
    device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    mut type_bits: u32,
    properties: vk::MemoryPropertyFlags,
) -> u32 {
    for i in 0..device_memory_properties.memory_type_count {
        if (type_bits & 1) == 1
            && (device_memory_properties.memory_types[i as usize].property_flags & properties)
                == properties
        {
            return i;
        }
        type_bits >>= 1;
    }
    0
}
