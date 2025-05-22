use std::sync::{Arc, RwLock};

use egui_winit_vulkano::{Gui, GuiConfig};
use glam::Vec3;
use vulkano::{
    Version,
    command_buffer::allocator::StandardCommandBufferAllocator,
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{DeviceExtensions, DeviceFeatures},
    format::Format,
    image::{Image, ImageCreateInfo, ImageType, ImageUsage, view::ImageView},
    instance::{
        InstanceCreateInfo, InstanceExtensions,
        debug::{
            DebugUtilsMessageSeverity, DebugUtilsMessageType, DebugUtilsMessengerCallback,
            DebugUtilsMessengerCreateInfo,
        },
    },
    memory::allocator::{AllocationCreateInfo, MemoryAllocator},
    swapchain::{ColorSpace, Surface},
};
use vulkano_util::{
    context::{VulkanoConfig, VulkanoContext},
    window::{VulkanoWindows, WindowDescriptor},
};
use winit::{
    application::ApplicationHandler, event::WindowEvent, raw_window_handle::HasDisplayHandle,
};

use crate::{
    gui::GuiState,
    raytracer::{Camera, Model, PerspectiveCamera, Scene, Vk},
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

    /// The image view for rendering the scene separate from the GUI.
    scene_image_view: Option<Arc<ImageView>>,

    /// The scene to render.
    scene: Option<Scene>,

    /// The egui/winit/vulkano wrapper.
    gui: Option<Gui>,

    /// Handles GUI statement management.
    gui_state: Option<GuiState>,

    /// The current scene file being rendered. This will be used to track when egui File > Open
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
            scene_image_view: None,
            scene: None,
            gui: None,
            gui_state: None,
            vk,
            current_file_path: DEFAULT_ASSET_FILE_PATH.to_string(),
        }
    }

    /// Rebuild the image view used for rendering and update camera settings when window size changes.
    fn rebuild_scene_image(&mut self, window_size: [f32; 2], format: Format, usage: ImageUsage) {
        let scene_image_view = create_scene_image(
            self.context.memory_allocator().clone(),
            window_size,
            format,
            usage,
        );

        let gui = self.gui.as_mut().unwrap();

        self.gui_state
            .as_mut()
            .unwrap()
            .update_scene_image(gui, scene_image_view.clone());

        self.scene_image_view = Some(scene_image_view);

        if let Some(scene) = self.scene.as_mut() {
            scene.update_window_size(window_size);
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
                ci.image_format = Format::B8G8R8A8_UNORM;
                ci.image_color_space = ColorSpace::SrgbNonLinear;
                ci.image_usage = ImageUsage::STORAGE  // For scene
                    | ImageUsage::SAMPLED | ImageUsage::COLOR_ATTACHMENT; // For egui
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
        let scene_image_view = create_scene_image(
            self.vk.memory_allocator.clone(),
            window_size,
            renderer.swapchain_format(),
            renderer.swapchain_image_view().usage(),
        );

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

        // Create the raytracing pipeline
        let scene = Scene::new(self.vk.clone(), &models, camera).unwrap();
        self.scene = Some(scene);

        // Create the GUI.
        let mut gui = Gui::new(
            event_loop,
            renderer.surface(),
            self.vk.queue.clone(),
            renderer.swapchain_format(),
            GuiConfig::default(),
        );
        self.gui_state = Some(GuiState::new(
            &mut gui,
            scene_image_view.clone(),
            DEFAULT_ASSET_FILE_PATH,
        ));
        self.gui = Some(gui);

        self.scene_image_view = Some(scene_image_view);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let renderer = self.windows.get_renderer_mut(window_id).unwrap();
        let renderer_window_id = renderer.window().id();
        let renderer_window_size = renderer.window_size();
        let renderer_scale_factor = renderer.window().scale_factor();

        let swapchain_format = renderer.swapchain_format();
        let swapchain_usage = renderer.swapchain_image_view().usage();

        let scene = self.scene.as_mut().unwrap();
        let scene_image_view = self.scene_image_view.as_ref().unwrap();

        match event {
            WindowEvent::Resized(window_size) => {
                renderer.resize();
                self.rebuild_scene_image(window_size.into(), swapchain_format, swapchain_usage);
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                renderer.resize();
                self.rebuild_scene_image(renderer_window_size, swapchain_format, swapchain_usage);
            }
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                // Handle File > Open.
                let gui = self.gui.as_mut().unwrap();
                let gui_state = self.gui_state.as_mut().unwrap();

                let gui_state_file_path = gui_state.get_file_path();
                if self.current_file_path != gui_state_file_path {
                    match Model::load_obj(gui_state_file_path) {
                        Ok(models) => match scene.rebuild(&models) {
                            Ok(()) => {
                                self.current_file_path = gui_state_file_path.to_string();
                            }
                            Err(e) => {
                                println!("Unable to load file {}. {:?}", gui_state_file_path, e);
                                gui_state.set_file_path(&self.current_file_path);
                            }
                        },

                        Err(e) => {
                            println!("Error loading file {}. {e:?}", gui_state_file_path);
                            gui_state.set_file_path(&self.current_file_path);
                        }
                    }
                }

                // Set immediate UI in redraw here.
                gui.immediate_ui(|gui| {
                    let ctx = gui.context();
                    self.gui_state.as_mut().unwrap().layout(
                        ctx,
                        renderer_window_size,
                        renderer_scale_factor,
                    )
                });

                // Acquire swapchain future and render the scene overlayed with the GUI.
                match renderer.acquire(None, |_| {}) {
                    Ok(future) => {
                        // Render scene
                        let after_scene_render = scene.render(future, scene_image_view.clone());

                        // Render gui
                        let after_gui_render =
                            gui.draw_on_image(after_scene_render, renderer.swapchain_image_view());

                        // Present swapchain
                        renderer.present(after_gui_render, true);
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

        if window_id == renderer_window_id {
            // Update Egui integration so the UI works!
            let gui = self.gui.as_mut().unwrap();
            let _pass_events_to_game = !gui.update(&event);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        let renderer = self.windows.get_primary_renderer().unwrap();
        renderer.window().request_redraw();
    }
}

/// Create a new image view to render the scene and overlay the GUI.
fn create_scene_image(
    memory_allocator: Arc<dyn MemoryAllocator>,
    window_size: [f32; 2],
    format: Format,
    usage: ImageUsage,
) -> Arc<ImageView> {
    ImageView::new_default(
        Image::new(
            memory_allocator,
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format,
                usage,
                extent: [
                    window_size[0].round() as u32,
                    window_size[1].round() as u32,
                    1,
                ],
                array_layers: 1,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap(),
    )
    .unwrap()
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
