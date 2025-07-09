use std::sync::Arc;

use anyhow::Result;
use ash::vk;
use log::debug;

use crate::VulkanContext;

pub struct DescriptorSetLayout {
    context: Arc<VulkanContext>,
    pub layout: vk::DescriptorSetLayout,
}

impl DescriptorSetLayout {
    pub fn new(
        context: Arc<VulkanContext>,
        bindings: &[vk::DescriptorSetLayoutBinding],
        binding_flags: &[vk::DescriptorBindingFlags],
    ) -> Result<Self> {
        let mut binding_flags_info =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(binding_flags);

        let create_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(bindings)
            .push_next(&mut binding_flags_info);

        let layout = unsafe {
            context
                .device
                .create_descriptor_set_layout(&create_info, None)?
        };

        Ok(Self { context, layout })
    }

    pub fn get(&self) -> vk::DescriptorSetLayout {
        self.layout
    }
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        debug!("DescriptorSetLayout::drop()");
        unsafe {
            self.context.device.device_wait_idle().unwrap();

            self.context
                .device
                .destroy_descriptor_set_layout(self.layout, None);
        }
    }
}
