use std::sync::Arc;

use anyhow::Result;
use ash::vk::{self, Handle};
use log::debug;

use crate::VulkanContext;

pub const NO_FENCE: Fence = Fence {
    context: None,
    fence: vk::Fence::null(),
};

pub struct Fence {
    context: Option<Arc<VulkanContext>>,
    fence: vk::Fence,
}

impl Fence {
    pub fn new(context: Arc<VulkanContext>, signaled: bool) -> Result<Self> {
        let fence_info = vk::FenceCreateInfo {
            flags: if signaled {
                vk::FenceCreateFlags::SIGNALED
            } else {
                vk::FenceCreateFlags::empty()
            },
            ..Default::default()
        };

        let fence = unsafe { context.device.create_fence(&fence_info, None)? };
        Ok(Self {
            context: Some(context),
            fence,
        })
    }

    pub fn get(&self) -> vk::Fence {
        self.fence
    }

    pub fn wait_and_reset(&self) -> Result<()> {
        if !self.fence.is_null()
            && let Some(context) = self.context.as_ref()
        {
            unsafe {
                // Wait for the fence to be signaled (GPU finished)
                context
                    .device
                    .wait_for_fences(&[self.fence], true, u64::MAX)?;

                // Reset the fence to unsignaled state for reuse
                context.device.reset_fences(&[self.fence])?;
            }
        }
        Ok(())
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        debug!("Fence::drop()");
        if !self.fence.is_null()
            && let Some(context) = self.context.as_ref()
        {
            unsafe {
                context.device.device_wait_idle().unwrap();
                context.device.destroy_fence(self.fence, None);
            }
        }
    }
}
