use ash::vk;

#[derive(Debug)]
pub struct Descriptor {
    pub descriptor_type: vk::DescriptorType,
    pub descriptor_count: u32,
}

impl Descriptor {
    pub fn new(descriptor_type: vk::DescriptorType, descriptor_count: u32) -> Self {
        Self {
            descriptor_type,
            descriptor_count,
        }
    }
}

impl From<Descriptor> for vk::DescriptorPoolSize {
    fn from(value: Descriptor) -> Self {
        vk::DescriptorPoolSize {
            ty: value.descriptor_type,
            descriptor_count: value.descriptor_count,
        }
    }
}

impl From<&Descriptor> for vk::DescriptorPoolSize {
    fn from(value: &Descriptor) -> Self {
        vk::DescriptorPoolSize {
            ty: value.descriptor_type,
            descriptor_count: value.descriptor_count,
        }
    }
}
