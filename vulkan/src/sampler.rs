use std::sync::Arc;

use anyhow::Result;
use ash::vk;

use crate::VulkanContext;

pub struct Sampler {
    pub sampler: vk::Sampler,

    context: Arc<VulkanContext>,
}

impl Sampler {
    pub fn new(context: Arc<VulkanContext>) -> Result<Self> {
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT);

        let sampler = unsafe { context.device.create_sampler(&sampler_info, None)? };
        Ok(Self { context, sampler })
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe {
            self.context.device.destroy_sampler(self.sampler, None);
        }
    }
}
