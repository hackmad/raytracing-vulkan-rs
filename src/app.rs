use crate::raytracer::{Camera, LightPropertyData, Model, PerspectiveCamera, Scene, Vk};
use glam::Vec3;
use std::sync::{Arc, RwLock};
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
    event::{ElementState, KeyEvent, WindowEvent},
    keyboard::{Key, NamedKey},
    raw_window_handle::HasDisplayHandle,
};

const DEFAULT_ASSET_FILE_PATH: &str = "assets/obj/sphere-on-plane.obj";
const INITIAL_WIDTH: u32 = 1024;
const INITIAL_HEIGHT: u32 = 576;

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

    /// The current scene file being rendered. This will be used to track egui File > Open
    /// will result in rebuilding a scene.
    current_file_path: String,
}

impl App {
    pub fn new(event_loop: &impl HasDisplayHandle, enable_debug_logging: bool) -> Self {
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
            current_file_path: DEFAULT_ASSET_FILE_PATH.to_string(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        // Create a new window and renderer. Note that fractional scaling in the OS will give a scaled width/height.
        self.windows.create_window(
            event_loop,
            &self.context,
            &WindowDescriptor {
                title: "Raytracing - Vulkan".to_string(),
                width: INITIAL_WIDTH as f32,
                height: INITIAL_HEIGHT as f32,
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

        println!("Swapchain image format: {:?}", renderer.swapchain_format());

        // Create storage image for rendering and display.
        let window_size = renderer.window_size();

        // Load models.
        let models = Model::load_obj(&self.current_file_path).unwrap();

        // Create camera.
        let camera: Arc<RwLock<dyn Camera>> = Arc::new(RwLock::new(PerspectiveCamera::new(
            Vec3::new(4.5, 3.0, -3.5),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            0.01,
            100.0,
            window_size[0] as u32,
            window_size[1] as u32,
        )));

        // Create lights.
        let lights = [
            LightPropertyData::new_spot(4.0, [3.0, 3.0, 0.0]),
            LightPropertyData::new_directional(1.0, [-3.0, 3.0, 0.0]),
        ];

        // Create the raytracing pipeline
        let scene = Scene::new(self.vk.clone(), &models, camera, &lights, window_size).unwrap();
        self.scene = Some(scene);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let renderer = self.windows.get_renderer_mut(window_id).unwrap();
        let window_size = renderer.window_size();
        let scene = self.scene.as_mut().unwrap();

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
                    println!("Escape key was pressed; stopping.");
                    event_loop.exit();
                }
                Key::Character("o") => {
                    // Handle File > Open.
                    let current_dir =
                        std::env::current_dir().expect("Unable to get current directory.");

                    let fd = rfd::FileDialog::new()
                        .set_directory(current_dir)
                        .add_filter("Wavefront (.obj)", &["obj"]);

                    if let Some(path) = fd.pick_file() {
                        let selected_path = path.display().to_string();

                        if self.current_file_path != selected_path {
                            match Model::load_obj(&selected_path) {
                                Ok(models) => match scene.rebuild(&models, window_size) {
                                    Ok(()) => {
                                        self.current_file_path = selected_path;
                                    }
                                    Err(e) => {
                                        println!("Unable to load file {}. {:?}", selected_path, e);
                                        self.current_file_path = selected_path;
                                    }
                                },

                                Err(e) => {
                                    println!("Error loading file {}. {e:?}", selected_path);
                                }
                            }
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
                        println!("Failed to acquire swapchain future: {}", e);
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

                    println!(
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
