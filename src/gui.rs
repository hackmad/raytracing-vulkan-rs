use std::sync::Arc;

use egui_winit_vulkano::{
    Gui,
    egui::{self, load::SizedTexture},
};
use glam::f64;
use vulkano::image::view::ImageView;

pub struct GuiState {
    scene_texture_id: egui::TextureId,
}

impl GuiState {
    pub fn new(gui: &mut Gui, scene_image: Arc<ImageView>) -> Self {
        GuiState {
            scene_texture_id: get_scene_texture_id(gui, scene_image),
        }
    }

    pub fn update_scene_image(&mut self, gui: &mut Gui, scene_image: Arc<ImageView>) {
        self.scene_texture_id = get_scene_texture_id(gui, scene_image);
    }

    pub fn layout(
        &mut self,
        egui_context: egui::Context,
        window_size: [f32; 2],
        scale_factor: f64,
    ) {
        let GuiState {
            scene_texture_id, ..
        } = self;

        egui_context.set_visuals(egui::Visuals::dark());

        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(&egui_context, |ui| {
                ui.image(egui::ImageSource::Texture(SizedTexture::new(
                    *scene_texture_id,
                    [
                        (window_size[0] as f64 / scale_factor) as f32,
                        (window_size[1] as f64 / scale_factor) as f32,
                    ],
                )));
            });

        egui::Window::new("Options")
            .resizable(false)
            .show(&egui_context, |ui| {
                ui.heading("Hello Tree");
            });
    }
}

fn get_scene_texture_id(gui: &mut Gui, scene_image: Arc<ImageView>) -> egui::TextureId {
    gui.register_user_image_view(scene_image, Default::default())
}
