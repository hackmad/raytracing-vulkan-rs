use std::{path::PathBuf, sync::Arc};

use anyhow::Result;
use log::{error, info};
use raytracer::{RenderEngine, RenderResult};
use scene_file::SceneFile;
use vulkan::VulkanContext;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::Key,
    window::{Window, WindowId},
};

const INITIAL_WINDOW_SIZE: [f32; 2] = [1024.0, 576.0];

/// Winit application.
pub struct App {
    /// The winit window.
    window: Option<Window>,

    /// Vulkan context.
    context: Option<Arc<VulkanContext>>,

    /// The render engine.
    render_engine: Option<RenderEngine>,

    /// The current scene file being rendered.
    current_file_path: String,

    /// This will be used to track egui File > Open will result in a new scene being loaded.
    new_file_path: Option<String>,

    /// Recreate the swapchain.
    recreate_swapchain: bool,
}

impl App {
    pub fn new(initial_file_path: &str) -> Result<Self> {
        Ok(Self {
            window: None,
            context: None,
            render_engine: None,
            current_file_path: initial_file_path.to_string(),
            new_file_path: None,
            recreate_swapchain: false,
        })
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Load scene file.
        let scene_file = SceneFile::load_json(&self.current_file_path).unwrap();

        // Create a new window.
        let window_size = adjust_window_size(INITIAL_WINDOW_SIZE, scene_file.render.aspect_ratio);

        let app_name = "Raytracing - Vulkan";

        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title(app_name)
                    .with_inner_size(LogicalSize::new(window_size[0], window_size[1])),
            )
            .expect("Failed to create window");

        let context = Arc::new(VulkanContext::new(app_name, &window).unwrap());

        // Create the render engine.
        let render_engine =
            RenderEngine::new(context.clone(), &scene_file, &window, &window_size).unwrap();

        self.window = Some(window);
        self.render_engine = Some(render_engine);
        self.context = Some(context);
        self.recreate_swapchain = false;
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                info!("The close button was pressed; stopping");
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
                Key::Character("q") => {
                    info!("Q was pressed; stopping.");
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

            WindowEvent::Resized(window_size) => {
                if let Some(render_engine) = self.render_engine.as_mut() {
                    render_engine
                        .update_window_size([window_size.width as f32, window_size.height as f32])
                        .unwrap();
                    self.recreate_swapchain = true;
                }
            }

            WindowEvent::ScaleFactorChanged { .. } => {
                if let Some(render_engine) = self.render_engine.as_mut()
                    && let Some(window) = &self.window
                {
                    let window_size = window.inner_size();

                    render_engine
                        .update_window_size([window_size.width as f32, window_size.height as f32])
                        .unwrap();

                    self.recreate_swapchain = true;
                }
            }

            WindowEvent::RedrawRequested => {
                if let Some(render_engine) = self.render_engine.as_mut()
                    && !self.recreate_swapchain
                {
                    match render_engine.render() {
                        Ok(RenderResult::RecreateSwapchain) => {
                            self.recreate_swapchain = true;
                        }

                        Ok(RenderResult::Done) => {
                            // Nothing to do
                        }

                        Err(e) => error!("failed to render. {e:?}"),
                    };
                }
            }

            _ => (),
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }

    fn new_events(&mut self, _event_loop: &ActiveEventLoop, _cause: winit::event::StartCause) {
        if let Some(window) = self.window.as_ref() {
            // Handle swapchain recreation.
            if self.recreate_swapchain
                && let Some(render_engine) = self.render_engine.as_mut()
            {
                render_engine
                    .recreate_swapchain(window)
                    .expect("failed to recreate swapchain");

                self.recreate_swapchain = false;

                window.request_redraw();
            }

            // Handle loading a new scene.
            if let Some(new_path) = self.new_file_path.take()
                && let Ok(scene_file) = SceneFile::load_json(&new_path)
                && let Some(context) = self.context.as_ref()
            {
                let window_size = window.inner_size();
                let size = [window_size.width as f32, window_size.height as f32];

                // Drop the old engine and create a new one.
                self.render_engine = None;
                self.render_engine = Some(
                    RenderEngine::new(context.clone(), &scene_file, window, &size)
                        .expect("failed to create render engine"),
                );

                self.current_file_path = new_path;

                window.request_redraw();
            }
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
