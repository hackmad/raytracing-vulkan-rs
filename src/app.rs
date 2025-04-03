use std::sync::{Arc, RwLock};

use egui_winit_vulkano::{Gui, GuiConfig};
use glam::Vec3;
use vulkano::{
    Version,
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{DeviceExtensions, DeviceFeatures},
    format::Format,
    image::{Image, ImageCreateInfo, ImageType, ImageUsage, view::ImageView},
    instance::{InstanceCreateInfo, InstanceExtensions},
    memory::allocator::{AllocationCreateInfo, MemoryAllocator},
    swapchain::Surface,
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
    raytracer::{Camera, Model, PerspectiveCamera, Scene},
};

const INITIAL_WIDTH: u32 = 1024;
const INITIAL_HEIGHT: u32 = 576;

pub struct App {
    context: VulkanoContext,
    windows: VulkanoWindows,
    scene_image: Option<Arc<ImageView>>,
    scene: Option<Scene>,
    gui: Option<Gui>,
    gui_state: Option<GuiState>,
}

impl App {
    pub fn new(event_loop: &impl HasDisplayHandle) -> Self {
        let required_extensions = Surface::required_extensions(event_loop)
            .expect("Failed to get required extensions to create a surface");

        // Vulkano context
        let context = VulkanoContext::new(VulkanoConfig {
            instance_create_info: InstanceCreateInfo {
                #[cfg(target_vendor = "apple")]
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                application_version: Version::V1_3,
                enabled_extensions: InstanceExtensions {
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

        // Scene view
        Self {
            context,
            windows,
            scene_image: None,
            scene: None,
            gui: None,
            gui_state: None,
        }
    }

    fn rebuild_scene_image(&mut self, window_size: [f32; 2]) {
        let scene_image = create_scene_image(self.context.memory_allocator().clone(), window_size);

        let gui = self.gui.as_mut().unwrap();

        self.gui_state
            .as_mut()
            .unwrap()
            .update_scene_image(gui, scene_image.clone());

        self.scene_image = Some(scene_image);

        if let Some(scene) = self.scene.as_mut() {
            scene.update_window_size(window_size);
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        // Note that fractional scaling in the OS will give a scaled width/height.
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
                ci.min_image_count = ci.min_image_count.max(2);
            },
        );

        let device = self.context.device();

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let renderer = self
            .windows
            .get_primary_renderer_mut()
            .expect("Failed to get primary renderer");

        let queue = renderer.graphics_queue();

        // Create storage image for rendering and display.
        let window_size = renderer.window_size();
        let scene_image = create_scene_image(self.context.memory_allocator().clone(), window_size);

        // Load models.
        let models = Model::load_obj("assets/obj/box.obj").unwrap();

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
        self.scene = Some(Scene::new(
            device.clone(),
            queue.clone(),
            self.context.memory_allocator().clone(),
            descriptor_set_allocator,
            &models[0], // TODO handle multiple models.
            camera,
        ));

        // Create gui
        let mut gui = Gui::new(
            event_loop,
            renderer.surface(),
            queue.clone(),
            renderer.swapchain_format(),
            GuiConfig::default(),
        );
        self.gui_state = Some(GuiState::new(&mut gui, scene_image.clone()));
        self.gui = Some(gui);

        self.scene_image = Some(scene_image);
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

        let scene = self.scene.as_mut().unwrap();
        let scene_image = self.scene_image.as_ref().unwrap();

        match event {
            WindowEvent::Resized(window_size) => {
                renderer.resize();
                self.rebuild_scene_image(window_size.into());
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                renderer.resize();
                self.rebuild_scene_image(renderer_window_size);
            }
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                let gui = self.gui.as_mut().unwrap();

                // Set immediate UI in redraw here
                gui.immediate_ui(|gui| {
                    let ctx = gui.context();
                    self.gui_state.as_mut().unwrap().layout(
                        ctx,
                        renderer_window_size,
                        renderer_scale_factor,
                    )
                });

                // Acquire swapchain future
                match renderer.acquire(Some(std::time::Duration::from_millis(10)), |_| {}) {
                    Ok(future) => {
                        // Render scene
                        let after_scene_render = scene.render(future, scene_image.clone());

                        // Render gui
                        let after_gui_render =
                            gui.draw_on_image(after_scene_render, renderer.swapchain_image_view());

                        // Present swapchain
                        renderer.present(after_gui_render, true);
                    }
                    Err(vulkano::VulkanError::OutOfDate) => {
                        renderer.resize();
                    }
                    Err(e) => panic!("Failed to acquire swapchain future: {}", e),
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

fn create_scene_image(
    memory_allocator: Arc<dyn MemoryAllocator>,
    window_size: [f32; 2],
) -> Arc<ImageView> {
    ImageView::new_default(
        Image::new(
            memory_allocator,
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::B8G8R8A8_UNORM,
                extent: [
                    window_size[0].round() as u32,
                    window_size[1].round() as u32,
                    1,
                ],
                array_layers: 1,
                usage: ImageUsage::STORAGE | // This is for the raytracer's descriptor set layout
                        ImageUsage::SAMPLED | ImageUsage::COLOR_ATTACHMENT, // These are for egui
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap(),
    )
    .unwrap()
}
