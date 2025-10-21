use std::sync::{Arc, LazyLock};

use anyhow::Result;
use ash::vk::{self, Handle};
use log::debug;

use crate::VulkanContext;

pub static NO_FENCE: LazyLock<Fence> = LazyLock::new(|| Fence {
    name: "NO_FENCE".to_string(),
    context: None,
    fence: vk::Fence::null(),
});

pub struct Fence {
    name: String,
    context: Option<Arc<VulkanContext>>,
    fence: vk::Fence,
}

impl Fence {
    pub fn new(context: Arc<VulkanContext>, name: &str, signaled: bool) -> Result<Self> {
        let fence_info = vk::FenceCreateInfo {
            flags: if signaled {
                vk::FenceCreateFlags::SIGNALED
            } else {
                vk::FenceCreateFlags::empty()
            },
            ..Default::default()
        };

        debug!("Creating fence {}", name);
        let fence = unsafe { context.device.create_fence(&fence_info, None)? };
        Ok(Self {
            name: name.to_string(),
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
                debug!("Fence::wait_and_reset(): waiting for {}", self.name);
                context
                    .device
                    .wait_for_fences(&[self.fence], true, u64::MAX)?;

                // Reset the fence to unsignaled state for reuse
                debug!("Fence::wait_and_reset(): resetting {}", self.name);
                context.device.reset_fences(&[self.fence])?;
            }
        } else {
            debug!("Fence::wait_and_reset(): ignoring null fence {}", self.name);
        }
        Ok(())
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        if !self.fence.is_null()
            && let Some(context) = self.context.as_ref()
        {
            debug!("Fence::drop(): destroying {}", self.name);
            unsafe {
                context.device.device_wait_idle().unwrap();
                context.device.destroy_fence(self.fence, None);
            }
        } else {
            debug!("Fence::drop(): ignoring null fence {}", self.name);
        }
    }
}
