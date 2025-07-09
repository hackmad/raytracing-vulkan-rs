use std::sync::Arc;

use anyhow::Result;
use ash::vk;
use log::debug;

use crate::{Buffer, Descriptor, DescriptorSetLayout, Sampler, VulkanContext, image::Image};

pub enum DescriptorSetBufferType {
    Uniform,
    Storage,
}

impl DescriptorSetBufferType {
    fn to_shader(&self) -> vk::DescriptorType {
        match self {
            Self::Uniform => vk::DescriptorType::UNIFORM_BUFFER,
            Self::Storage => vk::DescriptorType::STORAGE_BUFFER,
        }
    }
}

/// Stores a descriptor pool and descriptor set along with some data.
/// This way, the data's drop will run after descriptor set is dropped.
pub struct DescriptorSet<T> {
    pub set: vk::DescriptorSet,

    context: Arc<VulkanContext>,
    pool: vk::DescriptorPool,

    _data: T,
}

impl<T> DescriptorSet<T> {
    /// Create a new descriptor set. This will store the data so that the data's drop will run after
    /// descriptor set is dropped.
    fn new(
        context: Arc<VulkanContext>,
        pool: vk::DescriptorPool,
        set: vk::DescriptorSet,
        data: T,
    ) -> Self {
        Self {
            context,
            pool,
            set,
            _data: data,
        }
    }
}

impl<T> Drop for DescriptorSet<T> {
    fn drop(&mut self) {
        debug!("DescriptorSet::drop()");
        unsafe {
            self.context.device.device_wait_idle().unwrap();

            self.context
                .device
                .free_descriptor_sets(self.pool, &[self.set])
                .unwrap();

            self.context.device.destroy_descriptor_pool(self.pool, None);
        }
    }
}

fn new_ds(
    context: Arc<VulkanContext>,
    descriptor_set_layout: &DescriptorSetLayout,
    descriptors: &[Descriptor],
    variable_descriptor_count: u32, // Use > 0 for variable descriptors.
) -> Result<(vk::DescriptorPool, vk::DescriptorSet)> {
    let descriptor_sizes: Vec<_> = descriptors
        .iter()
        .map(vk::DescriptorPoolSize::from)
        .collect();

    let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
        .pool_sizes(&descriptor_sizes)
        .max_sets(1);

    let descriptor_pool = unsafe {
        context
            .device
            .create_descriptor_pool(&descriptor_pool_info, None)?
    };

    let layouts = [descriptor_set_layout.get()];

    let variable_descriptor_counts = [variable_descriptor_count];

    let mut variable_descriptor_count_alloc_info =
        vk::DescriptorSetVariableDescriptorCountAllocateInfo::default()
            .descriptor_counts(&variable_descriptor_counts);

    let mut alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&layouts);

    if variable_descriptor_count > 0 {
        alloc_info = alloc_info.push_next(&mut variable_descriptor_count_alloc_info);
    }

    let descriptor_set = unsafe { context.device.allocate_descriptor_sets(&alloc_info)? }[0];

    Ok((descriptor_pool, descriptor_set))
}

pub fn new_tlas_ds(
    context: Arc<VulkanContext>,
    descriptor_set_layout: &DescriptorSetLayout,
    data: vk::AccelerationStructureKHR,
) -> Result<DescriptorSet<vk::AccelerationStructureKHR>> {
    let descriptors = [Descriptor::new(
        vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
        1,
    )];

    let (descriptor_pool, descriptor_set) =
        new_ds(context.clone(), descriptor_set_layout, &descriptors, 0)?;

    let accel_structs = [data];
    let mut accel_info = vk::WriteDescriptorSetAccelerationStructureKHR::default()
        .acceleration_structures(&accel_structs);

    let descriptor_writes = [vk::WriteDescriptorSet::default()
        .dst_set(descriptor_set)
        .dst_binding(0)
        .dst_array_element(0)
        .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
        .descriptor_count(1)
        .push_next(&mut accel_info)];

    unsafe {
        context
            .clone()
            .device
            .update_descriptor_sets(&descriptor_writes, &[]);
    }

    Ok(DescriptorSet::new(
        context,
        descriptor_pool,
        descriptor_set,
        data,
    ))
}

pub fn new_storage_image_ds<'a>(
    context: Arc<VulkanContext>,
    descriptor_set_layout: &'a DescriptorSetLayout,
    data: &'a Image,
) -> Result<DescriptorSet<&'a Image>> {
    let descriptors = [Descriptor::new(vk::DescriptorType::STORAGE_IMAGE, 1)];

    let (descriptor_pool, descriptor_set) =
        new_ds(context.clone(), descriptor_set_layout, &descriptors, 0)?;

    let image_info = [vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::GENERAL)
        .image_view(data.image_view)];

    let descriptor_writes = [vk::WriteDescriptorSet::default()
        .dst_set(descriptor_set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .image_info(&image_info)];

    unsafe {
        context
            .clone()
            .device
            .update_descriptor_sets(&descriptor_writes, &[]);
    }

    Ok(DescriptorSet::new(
        context,
        descriptor_pool,
        descriptor_set,
        data,
    ))
}

pub fn new_buffer_ds(
    context: Arc<VulkanContext>,
    descriptor_set_layout: &DescriptorSetLayout,
    ty: DescriptorSetBufferType,
    data: Buffer,
) -> Result<DescriptorSet<Buffer>> {
    let descriptors = [Descriptor::new(ty.to_shader(), 1)];

    let (descriptor_pool, descriptor_set) =
        new_ds(context.clone(), descriptor_set_layout, &descriptors, 0)?;

    let buffer_info = [vk::DescriptorBufferInfo::default()
        .buffer(data.buffer)
        .offset(0)
        .range(vk::WHOLE_SIZE)];

    let descriptor_writes = [vk::WriteDescriptorSet::default()
        .dst_set(descriptor_set)
        .dst_binding(0)
        .descriptor_type(ty.to_shader())
        .descriptor_count(1)
        .dst_array_element(0)
        .buffer_info(&buffer_info)];

    unsafe {
        context
            .clone()
            .device
            .update_descriptor_sets(&descriptor_writes, &[]);
    }

    Ok(DescriptorSet::new(
        context,
        descriptor_pool,
        descriptor_set,
        data,
    ))
}

pub fn new_buffers_ds(
    context: Arc<VulkanContext>,
    descriptor_set_layout: &DescriptorSetLayout,
    ty: DescriptorSetBufferType,
    data: Vec<Buffer>,
) -> Result<DescriptorSet<Vec<Buffer>>> {
    let descriptors = [Descriptor::new(ty.to_shader(), data.len() as _)];

    let (descriptor_pool, descriptor_set) =
        new_ds(context.clone(), descriptor_set_layout, &descriptors, 0)?;

    let buffer_infos: Vec<_> = data
        .iter()
        .map(|buffer| {
            vk::DescriptorBufferInfo::default()
                .buffer(buffer.buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE)
        })
        .collect();

    let descriptor_writes = buffer_infos
        .iter()
        .enumerate()
        .map(|(binding, buffer_info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(binding as _)
                .descriptor_type(ty.to_shader())
                .descriptor_count(1)
                .dst_array_element(0)
                .buffer_info(std::slice::from_ref(buffer_info))
        })
        .collect::<Vec<_>>();

    unsafe {
        context
            .clone()
            .device
            .update_descriptor_sets(&descriptor_writes, &[]);
    }

    Ok(DescriptorSet::new(
        context,
        descriptor_pool,
        descriptor_set,
        data,
    ))
}

pub fn new_sampler_and_textures_ds<I>(
    context: Arc<VulkanContext>,
    descriptor_set_layout: &DescriptorSetLayout,
    sampler: Sampler,
    texture_image_views: I,
) -> Result<DescriptorSet<Sampler>>
where
    I: IntoIterator<Item = vk::ImageView> + ExactSizeIterator,
{
    let image_count = texture_image_views.len() as u32;

    let descriptors = vec![
        Descriptor::new(vk::DescriptorType::SAMPLER, 1),
        Descriptor::new(vk::DescriptorType::SAMPLED_IMAGE, image_count.max(1)),
    ];

    let (descriptor_pool, descriptor_set) = new_ds(
        context.clone(),
        descriptor_set_layout,
        &descriptors,
        image_count,
    )?;

    let sampler_info = [vk::DescriptorImageInfo {
        sampler: sampler.sampler,
        image_view: vk::ImageView::null(), // not used for sampler
        image_layout: vk::ImageLayout::UNDEFINED, // not used for sampler
    }];

    let mut descriptor_writes = vec![
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .image_info(&sampler_info),
    ];

    let image_infos: Vec<_> = texture_image_views
        .into_iter()
        .map(|image_view| vk::DescriptorImageInfo {
            sampler: vk::Sampler::null(),
            image_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        })
        .collect();

    if image_count > 0 {
        descriptor_writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(&image_infos),
        );
    }

    unsafe {
        context
            .clone()
            .device
            .update_descriptor_sets(&descriptor_writes, &[]);
    }

    Ok(DescriptorSet::new(
        context,
        descriptor_pool,
        descriptor_set,
        sampler,
    ))
}
