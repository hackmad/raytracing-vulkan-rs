use std::{
    borrow::Cow,
    collections::HashSet,
    ffi::{CStr, CString, c_char},
};

use anyhow::{Context, Result};
use ash::{
    ext::debug_utils,
    khr::{surface, swapchain},
    vk,
};
use log::{debug, error, info, warn};
use winit::{
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::Window,
};

/// Our own Vulkan context. Wraps some common resources we will want to use.
pub struct VulkanContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub device: ash::Device,

    pub debug_utils_instance: debug_utils::Instance,
    pub debug_utils_loader: debug_utils::Device,
    pub surface_loader: surface::Instance,
    pub swapchain_loader: swapchain::Device,

    pub debug_callback: vk::DebugUtilsMessengerEXT,

    pub physical_device: vk::PhysicalDevice,
    pub physical_device_properties: vk::PhysicalDeviceProperties,
    pub device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub queue_family_index: u32,

    pub surface: vk::SurfaceKHR,

    pub command_pool: vk::CommandPool,

    /// Note this is maximum recursion depth for traceRays. This is different from the scene file recursion depth
    /// which is accumulating radiance by successively calling traceRays as many times as we need in batches.
    pub rt_pipeline_max_recursion_depth: u32,
}

impl VulkanContext {
    pub fn new(app_name: &str, window: &Window) -> Result<Self> {
        let entry = unsafe { ash::Entry::load()? };
        let instance = create_instance(app_name, &entry, window)?;

        let (debug_callback, debug_utils_instance) = setup_debug_callback(&entry, &instance)?;

        let display_handle = window.display_handle()?.as_raw();
        let window_handle = window.window_handle()?.as_raw();

        let surface_loader = surface::Instance::new(&entry, &instance);
        let surface = unsafe {
            ash_window::create_surface(&entry, &instance, display_handle, window_handle, None)?
        };

        let (physical_device, queue_family_index) = get_physical_device_and_queue_family_index(
            &instance,
            &[
                ash::khr::acceleration_structure::NAME,
                ash::khr::deferred_host_operations::NAME,
                ash::khr::ray_tracing_pipeline::NAME,
            ],
        )?;

        let device = create_device(&instance, physical_device, queue_family_index)?;

        let swapchain_loader = swapchain::Device::new(&instance, &device);

        let command_pool = create_command_pool(&device, queue_family_index)?;

        let device_memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let physical_device_properties =
            unsafe { instance.get_physical_device_properties(physical_device) };

        let mut ray_tracing_pipeline_props =
            vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
        let mut props2 =
            vk::PhysicalDeviceProperties2::default().push_next(&mut ray_tracing_pipeline_props);
        unsafe {
            instance.get_physical_device_properties2(physical_device, &mut props2);
        }

        let debug_utils_loader = debug_utils::Device::new(&instance, &device);

        // cleanup(); the 'drop' function will take care of it.
        Ok(Self {
            entry,
            instance,
            device,
            queue_family_index,
            physical_device,
            physical_device_properties,
            device_memory_properties,
            surface,
            surface_loader,
            swapchain_loader,
            command_pool,
            debug_callback,
            debug_utils_loader,
            debug_utils_instance,
            rt_pipeline_max_recursion_depth: ray_tracing_pipeline_props.max_ray_recursion_depth,
        })
    }

    pub fn get_graphics_queue(&self) -> vk::Queue {
        unsafe { self.device.get_device_queue(self.queue_family_index, 0) }
    }

    pub fn set_debug_utils_object_name<T: vk::Handle>(
        &self,
        object_handle: T,
        object_type: vk::ObjectType,
        object_name: &str,
    ) -> Result<()> {
        let name_cstr = CString::new(object_name).expect("wrong string parameter");

        let name_info = vk::DebugUtilsObjectNameInfoEXT {
            s_type: vk::StructureType::DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
            p_next: std::ptr::null(),
            object_type,
            object_handle: object_handle.as_raw(),
            p_object_name: name_cstr.as_ptr(),
            ..Default::default()
        };

        unsafe {
            self.debug_utils_loader
                .set_debug_utils_object_name(&name_info)?
        };

        Ok(())
    }

    #[allow(dead_code)]
    fn is_format_supported_for_storage_image(&self, format: vk::Format) -> bool {
        let format_info = vk::PhysicalDeviceImageFormatInfo2::default()
            .format(format)
            .ty(vk::ImageType::TYPE_2D)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::STORAGE)
            .flags(vk::ImageCreateFlags::empty());

        let mut image_format_properties = vk::ImageFormatProperties2::default();

        unsafe {
            self.instance.get_physical_device_image_format_properties2(
                self.physical_device,
                &format_info,
                &mut image_format_properties,
            )
        }
        .is_ok()
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();

            self.device.destroy_command_pool(self.command_pool, None);

            self.device.destroy_device(None);

            self.surface_loader.destroy_surface(self.surface, None);

            self.debug_utils_instance
                .destroy_debug_utils_messenger(self.debug_callback, None);

            self.instance.destroy_instance(None);
        }
    }
}

fn create_command_pool(device: &ash::Device, queue_family_index: u32) -> Result<vk::CommandPool> {
    let pool_create_info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(queue_family_index);

    let pool = unsafe { device.create_command_pool(&pool_create_info, None)? };
    Ok(pool)
}

fn create_instance(app_name: &str, entry: &ash::Entry, window: &Window) -> Result<ash::Instance> {
    let layer_names = [c"VK_LAYER_KHRONOS_validation"];
    let layers_names_raw: Vec<*const c_char> = layer_names
        .iter()
        .map(|raw_name| raw_name.as_ptr())
        .collect();

    let mut extension_names =
        ash_window::enumerate_required_extensions(window.display_handle()?.as_raw())?.to_vec();

    extension_names.push(ash::ext::debug_utils::NAME.as_ptr());

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        extension_names.push(ash::khr::portability_enumeration::NAME.as_ptr());
        // Enabling this extension is a requirement when using `VK_KHR_portability_subset`
        extension_names.push(ash::khr::get_physical_device_properties2::NAME.as_ptr());
    }

    let app_name = CString::new(app_name)?;
    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_name(&app_name)
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::API_VERSION_1_3);

    let create_flags = if cfg!(any(target_os = "macos", target_os = "ios")) {
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        vk::InstanceCreateFlags::default()
    };

    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_layer_names(&layers_names_raw)
        .enabled_extension_names(&extension_names)
        .flags(create_flags);

    let instance = unsafe { entry.create_instance(&create_info, None)? };
    Ok(instance)
}

fn get_physical_device_and_queue_family_index(
    instance: &ash::Instance,
    extensions: &[&CStr],
) -> Result<(vk::PhysicalDevice, u32)> {
    unsafe { instance.enumerate_physical_devices() }?
        .into_iter()
        .find_map(|physical_device| {
            let has_all_extesions =
                unsafe { instance.enumerate_device_extension_properties(physical_device) }.map(
                    |exts| {
                        let set: HashSet<_> = exts
                            .iter()
                            .map(|ext| unsafe {
                                CStr::from_ptr(&ext.extension_name as *const c_char)
                            })
                            .collect();

                        extensions.iter().all(|ext| set.contains(ext))
                    },
                );
            if has_all_extesions != Ok(true) {
                return None;
            }

            let graphics_family =
                unsafe { instance.get_physical_device_queue_family_properties(physical_device) }
                    .into_iter()
                    .enumerate()
                    .find(|(_, device_properties)| {
                        device_properties.queue_count > 0
                            && device_properties
                                .queue_flags
                                .contains(vk::QueueFlags::GRAPHICS)
                    });

            graphics_family.map(|(i, _)| (physical_device, i as u32))
        })
        .context("Couldn't find suitable device.")
}

fn create_device(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
) -> Result<ash::Device> {
    let device_extension_names_raw = [
        ash::ext::scalar_block_layout::NAME.as_ptr(),
        ash::khr::acceleration_structure::NAME.as_ptr(),
        ash::khr::buffer_device_address::NAME.as_ptr(),
        ash::khr::deferred_host_operations::NAME.as_ptr(),
        ash::khr::get_memory_requirements2::NAME.as_ptr(),
        ash::khr::ray_tracing_pipeline::NAME.as_ptr(),
        ash::khr::ray_tracing_maintenance1::NAME.as_ptr(),
        ash::khr::spirv_1_4::NAME.as_ptr(),
        ash::khr::swapchain::NAME.as_ptr(),
        ash::khr::synchronization2::NAME.as_ptr(),
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        ash::khr::portability_subset::NAME.as_ptr(),
    ];

    // Required features.
    let features = vk::PhysicalDeviceFeatures {
        shader_int64: 1,
        ..Default::default()
    };

    let mut ray_tracing_pipeline_features =
        vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default().ray_tracing_pipeline(true);

    let mut accel_struct_features =
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default().acceleration_structure(true);

    let mut vulkan_1_2_features = vk::PhysicalDeviceVulkan12Features::default()
        .runtime_descriptor_array(true)
        .descriptor_binding_partially_bound(true)
        .descriptor_binding_variable_descriptor_count(true)
        .buffer_device_address(true);

    let mut features2 = vk::PhysicalDeviceFeatures2::default()
        .features(features)
        .push_next(&mut ray_tracing_pipeline_features)
        .push_next(&mut accel_struct_features)
        .push_next(&mut vulkan_1_2_features);

    let priorities = [1.0];

    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&priorities);

    let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(std::slice::from_ref(&queue_info))
        .enabled_extension_names(&device_extension_names_raw)
        .push_next(&mut features2);

    let device = unsafe { instance.create_device(physical_device, &device_create_info, None)? };
    Ok(device)
}

fn setup_debug_callback(
    entry: &ash::Entry,
    instance: &ash::Instance,
) -> Result<(vk::DebugUtilsMessengerEXT, debug_utils::Instance)> {
    let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(vulkan_debug_callback));

    let debug_utils_instance = debug_utils::Instance::new(entry, instance);

    let debug_callback =
        unsafe { debug_utils_instance.create_debug_utils_messenger(&debug_info, None)? };

    Ok((debug_callback, debug_utils_instance))
}

extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = unsafe { *p_callback_data };

    let message_id_number = callback_data.message_id_number;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        unsafe { CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy() }
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        unsafe { CStr::from_ptr(callback_data.p_message).to_string_lossy() }
    };

    let msg = format!("{message_type:?} [{message_id_name} ({message_id_number})] : {message}");

    match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => {
            info!("{msg}");
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
            error!("{msg}");
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
            warn!("{msg}");
        }
        _ => {
            debug!("{msg}");
        }
    }

    vk::FALSE
}
