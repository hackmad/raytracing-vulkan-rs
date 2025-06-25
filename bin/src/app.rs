use std::{path::PathBuf, sync::Arc};

use log::{debug, error, info};
use vulkano::{
    Version,
    command_buffer::allocator::StandardCommandBufferAllocator,
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{DeviceExtensions, DeviceFeatures},
    image::ImageUsage,
    instance::{
        InstanceCreateInfo, InstanceExtensions,
        debug::{
            DebugUtilsMessageSeverity, DebugUtilsMessageType, DebugUtilsMessengerCallback,
            DebugUtilsMessengerCreateInfo,
        },
    },
    swapchain::Surface,
};
use vulkano_util::{
    context::{VulkanoConfig, VulkanoContext},
    window::{VulkanoWindows, WindowDescriptor},
};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, KeyEvent, WindowEvent},
    keyboard::{Key, NamedKey},
    raw_window_handle::HasDisplayHandle,
};

use raytracer::{Scene, Vk};
use scene_file::SceneFile;

const INITIAL_WINDOW_SIZE: [f32; 2] = [1024.0, 576.0];

/// Winit application.
pub struct App {
    /// Vulkano context.
    context: VulkanoContext,

    /// Handles window management.
    windows: VulkanoWindows,

    /// Our own vulkano context.
    vk: Arc<Vk>,

    /// The scene to render.
    scene: Option<Scene>,

    /// The current scene file being rendered.
    current_file_path: String,

    /// This will be used to track egui File > Open will result in a new scene being loaded.
    new_file_path: Option<String>,
}

impl App {
    pub fn new(
        event_loop: &impl HasDisplayHandle,
        enable_debug_logging: bool,
        initial_file_path: &str,
    ) -> Self {
        // Use extension supporting the winit event loop.
        let required_extensions = Surface::required_extensions(event_loop)
            .expect("Failed to get required extensions to create a surface");

        // Vulkano context
        let context = VulkanoContext::new(VulkanoConfig {
            debug_create_info: setup_debug_callback(enable_debug_logging),
            instance_create_info: InstanceCreateInfo {
                #[cfg(target_vendor = "apple")]
                flags: vulkano::instance::InstanceCreateFlags::ENUMERATE_PORTABILITY,
                application_version: Version::V1_3,
                enabled_extensions: InstanceExtensions {
                    ext_debug_utils: true,
                    ext_debug_report: true,
                    ext_swapchain_colorspace: true,
                    ..required_extensions
                },
                ..Default::default()
            },
            device_extensions: DeviceExtensions {
                khr_acceleration_structure: true,
                khr_deferred_host_operations: true,
                khr_ray_tracing_pipeline: true,
                khr_ray_tracing_maintenance1: true,
                khr_swapchain: true,
                khr_synchronization2: true,
                ..DeviceExtensions::empty()
            },
            device_features: DeviceFeatures {
                acceleration_structure: true,
                buffer_device_address: true,
                descriptor_binding_variable_descriptor_count: true,
                ray_tracing_pipeline: true,
                runtime_descriptor_array: true,
                scalar_block_layout: true,
                shader_int64: true,
                synchronization2: true,
                ..Default::default()
            },
            print_device_name: true,
            ..Default::default()
        });

        // Vulkano windows
        let windows = VulkanoWindows::default();

        // Create some common alloctors we want to use.
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            context.device().clone(),
            Default::default(),
        ));

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            context.device().clone(),
            Default::default(),
        ));

        // Create our own Vulkan context.
        let vk = Arc::new(Vk {
            device: context.device().clone(),
            queue: context.graphics_queue().clone(),
            memory_allocator: context.memory_allocator().clone(),
            command_buffer_allocator,
            descriptor_set_allocator,
        });

        // Create the app with a default asset file loaded.
        Self {
            context,
            windows,
            scene: None,
            vk,
            current_file_path: initial_file_path.to_string(),
            new_file_path: None,
        }
    }
}

fn adjust_window_size(mut window_size: [f32; 2], aspect_ratio: f32) -> [f32; 2] {
    if window_size[0] > window_size[1] {
        window_size[0] = aspect_ratio * window_size[1];
    } else {
        window_size[1] = window_size[0] / aspect_ratio;
    }
    window_size
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        // Load scene file.
        let scene_file = SceneFile::load_json(&self.current_file_path).unwrap();

        let mut window_size =
            adjust_window_size(INITIAL_WINDOW_SIZE, scene_file.render.aspect_ratio);

        // Create a new window and renderer.
        self.windows.create_window(
            event_loop,
            &self.context,
            &WindowDescriptor {
                title: "Raytracing - Vulkan".to_string(),
                width: window_size[0],
                height: window_size[1],
                ..Default::default()
            },
            |ci| {
                ci.image_usage = ImageUsage::STORAGE;
                ci.min_image_count = ci.min_image_count.max(2);
            },
        );

        let renderer = self
            .windows
            .get_primary_renderer_mut()
            .expect("Failed to get primary renderer");

        info!("Swapchain image format: {:?}", renderer.swapchain_format());

        // Refetch window size from renderer because window creation will account for fractional scaling.
        window_size = renderer.window_size();

        // Create scene.
        let scene = Scene::new(self.vk.clone(), &scene_file, &window_size).unwrap();
        self.scene = Some(scene);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let renderer = self.windows.get_renderer_mut(window_id).unwrap();
        let scene = self.scene.as_mut().unwrap();

        // Handle loading a new scene before processing events.
        if let Some(new_scene_path) = &self.new_file_path {
            match SceneFile::load_json(new_scene_path) {
                Ok(scene_file) => {
                    // Resize the window based on initial dimensions and scene aspect ratio.
                    let mut window_size =
                        adjust_window_size(INITIAL_WINDOW_SIZE, scene_file.render.aspect_ratio);
                    let _ = renderer
                        .window()
                        .request_inner_size(LogicalSize::new(window_size[0], window_size[1]));

                    // Refetch window size from renderer because window creation will account for fractional scaling.
                    window_size = renderer.window_size();

                    match Scene::new(self.vk.clone(), &scene_file, &window_size) {
                        Ok(new_scene) => {
                            *scene = new_scene;
                            self.current_file_path = new_scene_path.clone();
                            self.new_file_path = None;
                        }
                        Err(e) => {
                            error!("Unable to load file {}. {:?}", new_scene_path, e);
                            self.new_file_path = None;
                        }
                    }
                }

                Err(e) => {
                    error!("Error loading file {}. {e:?}", new_scene_path);
                }
            }
        }

        match event {
            WindowEvent::Resized(window_size) => {
                scene.update_window_size([window_size.width as f32, window_size.height as f32]);
                renderer.resize();
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                scene.update_window_size(renderer.window_size());
                renderer.resize();
            }
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: key,
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => match key.as_ref() {
                Key::Named(NamedKey::Escape) => {
                    info!("Escape key was pressed; stopping.");
                    event_loop.exit();
                }
                Key::Character("o") => {
                    // Handle File > Open.
                    let current_file_path_buf = PathBuf::from(&self.current_file_path);
                    let current_dir_path = current_file_path_buf
                        .parent()
                        .expect("Unable to get current directory.");
                    let absolute_path = std::fs::canonicalize(current_dir_path)
                        .expect("Unable to get absolute path of current directory.");

                    let fd = rfd::FileDialog::new()
                        .set_directory(absolute_path)
                        .add_filter("JSON (.json)", &["json"]);

                    if let Some(path) = fd.pick_file() {
                        let selected_path = path.display().to_string();

                        if self.current_file_path != selected_path {
                            self.new_file_path = Some(selected_path);
                        }
                    }
                }
                _ => (),
            },
            WindowEvent::RedrawRequested => {
                // Acquire swapchain future and render the scene overlayed with the GUI.
                match renderer.acquire(None, |_| {}) {
                    Ok(future) => {
                        // Render scene
                        let after_scene_render =
                            scene.render(future, renderer.swapchain_image_view());

                        // Present swapchain
                        renderer.present(after_scene_render, true);
                    }
                    Err(vulkano::VulkanError::OutOfDate) => {
                        renderer.resize();
                    }
                    Err(e) => {
                        error!("Failed to acquire swapchain future: {}", e);
                        event_loop.exit();
                    }
                };
            }
            _ => (),
        }
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        let renderer = self.windows.get_primary_renderer().unwrap();
        renderer.window().request_redraw();
    }
}

/// Setup callback for logging debug information the GPU.
fn setup_debug_callback(enable_debug_logging: bool) -> Option<DebugUtilsMessengerCreateInfo> {
    let debug_callback = if enable_debug_logging {
        unsafe {
            Some(DebugUtilsMessengerCallback::new(
                |message_severity, message_type, callback_data| {
                    let severity = if message_severity.intersects(DebugUtilsMessageSeverity::ERROR)
                    {
                        "error"
                    } else if message_severity.intersects(DebugUtilsMessageSeverity::WARNING) {
                        "warning"
                    } else if message_severity.intersects(DebugUtilsMessageSeverity::INFO) {
                        "information"
                    } else if message_severity.intersects(DebugUtilsMessageSeverity::VERBOSE) {
                        "verbose"
                    } else {
                        panic!("no-impl");
                    };

                    let ty = if message_type.intersects(DebugUtilsMessageType::GENERAL) {
                        "general"
                    } else if message_type.intersects(DebugUtilsMessageType::VALIDATION) {
                        "validation"
                    } else if message_type.intersects(DebugUtilsMessageType::PERFORMANCE) {
                        "performance"
                    } else {
                        panic!("no-impl");
                    };

                    debug!(
                        "{} {} {}: {}",
                        callback_data.message_id_name.unwrap_or("unknown"),
                        ty,
                        severity,
                        callback_data.message
                    );
                },
            ))
        }
    } else {
        None
    };

    debug_callback.map(|callback| DebugUtilsMessengerCreateInfo {
        message_severity: DebugUtilsMessageSeverity::ERROR
            | DebugUtilsMessageSeverity::WARNING
            | DebugUtilsMessageSeverity::INFO
            | DebugUtilsMessageSeverity::VERBOSE,
        message_type: DebugUtilsMessageType::GENERAL
            | DebugUtilsMessageType::VALIDATION
            | DebugUtilsMessageType::PERFORMANCE,
        ..DebugUtilsMessengerCreateInfo::user_callback(callback)
    })
}
