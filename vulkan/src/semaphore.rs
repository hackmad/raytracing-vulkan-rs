use std::sync::Arc;

use anyhow::Result;
use ash::vk;

use crate::VulkanContext;

pub struct Semaphore {
    context: Arc<VulkanContext>,
    semaphore: vk::Semaphore,
}

impl Semaphore {
    pub fn new(context: Arc<VulkanContext>) -> Result<Self> {
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let semaphore = unsafe { context.device.create_semaphore(&semaphore_info, None)? };
        Ok(Self { context, semaphore })
    }

    pub fn get(&self) -> vk::Semaphore {
        self.semaphore
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        unsafe {
            self.context.device.destroy_semaphore(self.semaphore, None);
        }
    }
}
