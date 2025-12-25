use std::sync::{Arc, RwLock};

use anyhow::{Context, Result};
use log::debug;
use scene_file::SceneFile;
use vulkano::{format::Format, image::view::ImageView, sync::GpuFuture};

use crate::{Camera, Vk, create_camera, render_engine::RenderEngine};

/// Describes the scene for raytracing.
pub struct Scene {
    /// Vulkano conext.
    vk: Arc<Vk>,

    /// Camera.
    camera: Arc<RwLock<dyn Camera>>,

    /// The render engine to use.
    render_engine: Option<RenderEngine>,
}

impl Scene {
    /// Create a new scene from the given models and camera.
    pub fn new(
        vk: Arc<Vk>,
        scene_file: &SceneFile,
        window_size: &[f32; 2],
        swapchain_format: Format,
    ) -> Result<Self> {
        let render_camera = &scene_file.render.camera;

        let scene_camera = scene_file
            .cameras
            .iter()
            .find(|&cam| cam.get_name() == render_camera)
            .with_context(|| format!("Camera ${render_camera} is no specified in cameras"))?;
        debug!("{scene_camera:?}");

        let camera = create_camera(scene_camera, window_size[0] as u32, window_size[1] as u32);

        RenderEngine::new(vk.clone(), scene_file, window_size, swapchain_format).map(
            |render_engine| Scene {
                vk,
                render_engine: Some(render_engine),
                camera,
            },
        )
    }

    /// Updates the camera image size to match a new window size.
    ///
    /// # Panics
    ///
    /// - Panics if the render_engine fails to update image size.
    pub fn update_window_size(&mut self, window_size: [f32; 2]) {
        let mut camera = self.camera.write().unwrap();
        camera.update_image_size(window_size[0] as u32, window_size[1] as u32);

        if let Some(render_engine) = self.render_engine.as_mut() {
            render_engine
                .update_image_size(
                    self.vk.clone(),
                    window_size[0] as u32,
                    window_size[1] as u32,
                )
                .unwrap();
        }
    }

    /// Renders a scene to an image view after the given future completes. This will return a new
    /// future for the rendering operation.
    ///
    /// # Panics
    ///
    /// - Panics if the render_engine fails to create.
    pub fn render(
        &mut self,
        before_future: Box<dyn GpuFuture>,
        swapchain_image_view: Arc<ImageView>,
    ) -> Box<dyn GpuFuture> {
        if let Some(render_engine) = self.render_engine.as_mut() {
            render_engine.render(
                self.vk.clone(),
                before_future,
                swapchain_image_view,
                self.camera.clone(),
            )
        } else {
            // Do nothing.
            before_future
        }
    }
}
