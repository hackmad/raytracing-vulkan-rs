use std::sync::Arc;

use anyhow::{Result, anyhow};
use ash::vk;
use log::debug;
use winit::window::Window;

use crate::{Fence, Image, Semaphore, VulkanContext};

pub enum SwapchainNextImage {
    Acquired(usize),
    RecreateSwapchain,
}

pub struct SurfaceProperties {
    surface_format: vk::SurfaceFormatKHR,
    resolution: vk::Extent2D,
    capabilities: vk::SurfaceCapabilitiesKHR,
    pre_transform: vk::SurfaceTransformFlagsKHR,
}

impl SurfaceProperties {
    pub fn new(
        surface_format: vk::SurfaceFormatKHR,
        resolution: vk::Extent2D,
        capabilities: vk::SurfaceCapabilitiesKHR,
        pre_transform: vk::SurfaceTransformFlagsKHR,
    ) -> Self {
        Self {
            surface_format,
            resolution,
            capabilities,
            pre_transform,
        }
    }
}

pub struct Swapchain {
    /// Vulkan context.
    context: Arc<VulkanContext>,

    /// Swapchain
    swapchain: vk::SwapchainKHR,

    /// Swapchain images.
    swapchain_images: Vec<Image>,

    /// Is destroyed.
    is_destroyed: bool,
}

impl Swapchain {
    pub fn new(context: Arc<VulkanContext>, window: &Window) -> Result<Self> {
        let surface_properties = get_surface_properties(context.clone(), window)?;
        let swapchain = create_swapchain(context.clone(), &surface_properties)?;
        let swapchain_images =
            create_swapchain_images(context.clone(), swapchain, &surface_properties)?;

        Ok(Self {
            context,
            swapchain,
            swapchain_images,
            is_destroyed: false,
        })
    }

    pub fn get(&self) -> vk::SwapchainKHR {
        self.swapchain
    }

    pub fn get_image(&self, index: usize) -> &Image {
        assert!(
            index < self.swapchain_images.len(),
            "Swapchain index out of bounds"
        );
        &self.swapchain_images[index]
    }

    pub fn num_images(&self) -> usize {
        self.swapchain_images.len()
    }

    pub fn acquire_next_image(
        &self,
        timeout: u64,
        semaphore: &Semaphore,
        fence: &Fence,
    ) -> Result<SwapchainNextImage> {
        let result = unsafe {
            self.context.swapchain_loader.acquire_next_image(
                self.swapchain,
                timeout,
                semaphore.get(),
                fence.get(),
            )
        };

        match result {
            Ok((image_index, false)) => Ok(SwapchainNextImage::Acquired(image_index as usize)),
            Ok((_image_index, true)) => Ok(SwapchainNextImage::RecreateSwapchain),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                Ok(SwapchainNextImage::RecreateSwapchain)
            }
            Err(e) => Err(anyhow!("{e:?}")),
        }
    }

    pub fn destroy(&mut self) -> Result<()> {
        unsafe {
            debug!("Swapchain::destroy() - waiting for GPU to be available");
            self.context.device.device_wait_idle()?;

            debug!("Swapchain::destroy() - destroying swapchain image views");
            for image in self.swapchain_images.drain(..) {
                // We can do this because we created these and called Image::new().
                self.context
                    .device
                    .destroy_image_view(image.image_view, None);
            }

            debug!("Swapchain::destroy() - destroying swapchain");
            self.context
                .swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }

        self.is_destroyed = true;

        Ok(())
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        debug!("Swapchain::drop()");

        // This is for when application exits and drops the swapchain without us getting a chance
        // to call destroy.
        if !self.is_destroyed {
            self.destroy().unwrap();
        }
    }
}

pub fn create_swapchain(
    context: Arc<VulkanContext>,
    surface_properties: &SurfaceProperties,
) -> Result<vk::SwapchainKHR> {
    let present_modes = unsafe {
        context
            .surface_loader
            .get_physical_device_surface_present_modes(context.physical_device, context.surface)?
    };

    let present_mode = present_modes
        .iter()
        .cloned()
        .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
        .unwrap_or(vk::PresentModeKHR::FIFO);

    let mut desired_image_count = surface_properties.capabilities.min_image_count + 1;
    if surface_properties.capabilities.max_image_count > 0
        && desired_image_count > surface_properties.capabilities.max_image_count
    {
        desired_image_count = surface_properties.capabilities.max_image_count;
    }

    let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
        .surface(context.surface)
        .min_image_count(desired_image_count)
        .image_color_space(surface_properties.surface_format.color_space)
        .image_format(surface_properties.surface_format.format)
        .image_extent(surface_properties.resolution)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(surface_properties.pre_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .image_array_layers(1);

    let swapchain = unsafe {
        context
            .swapchain_loader
            .create_swapchain(&swapchain_create_info, None)?
    };

    Ok(swapchain)
}

pub fn create_swapchain_images(
    context: Arc<VulkanContext>,
    swapchain: vk::SwapchainKHR,
    surface_properties: &SurfaceProperties,
) -> Result<Vec<Image>> {
    let images = unsafe { context.swapchain_loader.get_swapchain_images(swapchain)? };

    let swapchain_images: Result<Vec<_>> = images
        .iter()
        .map(|&image| {
            let create_view_info = vk::ImageViewCreateInfo::default()
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(surface_properties.surface_format.format)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::R,
                    g: vk::ComponentSwizzle::G,
                    b: vk::ComponentSwizzle::B,
                    a: vk::ComponentSwizzle::A,
                })
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image(image);

            let image_view = unsafe {
                context
                    .device
                    .create_image_view(&create_view_info, None)
                    .map_err(|e| anyhow!("Failed to create image view. {e:?}"))
            };

            image_view.map(|iv| {
                Image::new(
                    context.clone(),
                    image,
                    iv,
                    surface_properties.resolution.width,
                    surface_properties.resolution.height,
                )
            })
        })
        .collect();

    swapchain_images
}

fn get_surface_properties(
    context: Arc<VulkanContext>,
    window: &Window,
) -> Result<SurfaceProperties> {
    let surface_format = unsafe {
        context
            .surface_loader
            .get_physical_device_surface_formats(context.physical_device, context.surface)?[0]
    };

    let surface_capabilities = unsafe {
        context
            .surface_loader
            .get_physical_device_surface_capabilities(context.physical_device, context.surface)?
    };

    let window_size = window.inner_size();
    let surface_resolution = match surface_capabilities.current_extent.width {
        u32::MAX => vk::Extent2D {
            width: window_size.width,
            height: window_size.height,
        },
        _ => surface_capabilities.current_extent,
    };

    let pre_transform = if surface_capabilities
        .supported_transforms
        .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
    {
        vk::SurfaceTransformFlagsKHR::IDENTITY
    } else {
        surface_capabilities.current_transform
    };

    Ok(SurfaceProperties::new(
        surface_format,
        surface_resolution,
        surface_capabilities,
        pre_transform,
    ))
}
