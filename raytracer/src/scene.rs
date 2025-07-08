use std::sync::{Arc, RwLock};

use anyhow::{Context, Result};
use log::debug;
use scene_file::SceneFile;
use vulkan::{Image, VulkanContext};

use crate::{Camera, RenderEngine, create_camera};

/// Describes the scene for raytracing.
pub struct Scene {
    /// Vulkan context.
    context: Arc<VulkanContext>,

    /// Camera.
    camera: Arc<RwLock<dyn Camera>>,

    /// Render engine.
    render_engine: RenderEngine,

    /// Image to render scene.
    render_image: Image,
}

impl Scene {
    /// Create a new scene from the given models and camera.
    pub fn new(
        context: Arc<VulkanContext>,
        scene_file: &SceneFile,
        window_size: &[f32; 2],
    ) -> Result<Self> {
        let render_camera = &scene_file.render.camera;

        let scene_camera = scene_file
            .cameras
            .iter()
            .find(|&cam| cam.get_name() == render_camera)
            .with_context(|| format!("Camera ${render_camera} is no specified in cameras"))?;
        debug!("{scene_camera:?}");

        let camera = create_camera(scene_camera, window_size[0] as u32, window_size[1] as u32);

        let render_engine = RenderEngine::new(context.clone(), scene_file, window_size)?;

        let render_image = Image::new_render_image(
            context.clone(),
            window_size[0] as u32,
            window_size[1] as u32,
        )?;

        Ok(Scene {
            context,
            camera,
            render_engine,
            render_image,
        })
    }

    /// Updates the camera image size to match a new window size.
    pub fn update_window_size(&mut self, window_size: [f32; 2]) -> Result<()> {
        let mut camera = self.camera.write().unwrap();
        camera.update_image_size(window_size[0] as u32, window_size[1] as u32);

        self.render_image = Image::new_render_image(
            self.context.clone(),
            window_size[0] as u32,
            window_size[1] as u32,
        )?;

        Ok(())
    }

    /// Renders a scene to an image view after the given future completes. This will return a new
    /// future for the rendering operation.
    pub fn render(&mut self) -> Result<()> {
        self.render_engine.render(
            self.context.clone(),
            &self.render_image,
            self.camera.clone(),
        )
    }
}
