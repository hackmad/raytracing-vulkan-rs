use std::sync::{Arc, RwLock};

use anyhow::{Context, Result};
use log::debug;
use vulkano::{image::view::ImageView, sync::GpuFuture};

use crate::{Camera, SceneFile, Vk, renderer::Renderer};

/// Describes the scene for raytracing.
pub struct Scene {
    /// Vulkano conext.
    vk: Arc<Vk>,

    /// Camera.
    camera: Arc<RwLock<dyn Camera>>,

    /// Vulkano resources specific to the rendering pipeline.
    resources: Option<Renderer>,
}

impl Scene {
    /// Create a new scene from the given models and camera.
    pub fn new(vk: Arc<Vk>, scene_file: &SceneFile, window_size: &[f32; 2]) -> Result<Self> {
        let render_camera = &scene_file.render.camera;

        let camera_type = scene_file
            .cameras
            .iter()
            .find(|&cam| cam.get_name() == render_camera)
            .with_context(|| format!("Camera ${render_camera} is no specified in cameras"))?;
        debug!("{camera_type:?}");

        let camera = camera_type.to_camera(window_size[0] as u32, window_size[1] as u32);

        Renderer::new(vk.clone(), scene_file, window_size).map(|resources| Scene {
            vk,
            resources: Some(resources),
            camera,
        })
    }

    /// Updates the camera image size to match a new window size.
    pub fn update_window_size(&mut self, window_size: [f32; 2]) {
        let mut camera = self.camera.write().unwrap();
        camera.update_image_size(window_size[0] as u32, window_size[1] as u32);
    }

    /// Renders a scene to an image view after the given future completes. This will return a new
    /// future for the rendering operation.
    ///
    /// # Panics
    ///
    /// - Panics if any Vulkan resources fail to create.
    pub fn render(
        &mut self,
        before_future: Box<dyn GpuFuture>,
        image_view: Arc<ImageView>,
    ) -> Box<dyn GpuFuture> {
        if let Some(resources) = self.resources.as_mut() {
            resources.render(
                self.vk.clone(),
                before_future,
                image_view,
                self.camera.clone(),
            )
        } else {
            // Do nothing.
            before_future
        }
    }
}
