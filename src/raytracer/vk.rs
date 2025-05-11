use std::sync::Arc;

use vulkano::{
    command_buffer::allocator::CommandBufferAllocator,
    descriptor_set::allocator::DescriptorSetAllocator,
    device::{Device, Queue},
    memory::allocator::MemoryAllocator,
};

// Our own Vulkan context. Wraps some common resources we will want to use.
pub struct Vk {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub memory_allocator: Arc<dyn MemoryAllocator>,
    pub command_buffer_allocator: Arc<dyn CommandBufferAllocator>,
    pub descriptor_set_allocator: Arc<dyn DescriptorSetAllocator>,
}
