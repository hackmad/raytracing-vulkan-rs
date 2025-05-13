use std::sync::Arc;

use egui_winit_vulkano::{
    Gui,
    egui::{self, Id, load::SizedTexture, panel::TopBottomSide},
};
use glam::f64;
use vulkano::image::view::ImageView;

/// Stores the GUI state.
pub struct GuiState {
    /// The texture identifier for the image used for rendering.
    scene_texture_id: egui::TextureId,

    /// The currently opened file path for the scene.
    file_path: String,
}

impl GuiState {
    /// Create a new state for the GUI.
    pub fn new(gui: &mut Gui, scene_image: Arc<ImageView>, file_path: &str) -> Self {
        GuiState {
            scene_texture_id: get_scene_texture_id(gui, scene_image),
            file_path: file_path.to_string(),
        }
    }

    /// Returns the current file path for the scene.
    pub fn get_file_path(&self) -> &str {
        &self.file_path
    }

    /// Sets the current file path for the scene.
    pub fn set_file_path(&mut self, path: &str) {
        self.file_path = path.to_string();
    }

    /// Update the image used for rendering.
    pub fn update_scene_image(&mut self, gui: &mut Gui, scene_image: Arc<ImageView>) {
        self.scene_texture_id = get_scene_texture_id(gui, scene_image);
    }

    /// Setup the GUI layout taking into account the window size and OS scaling factor.
    ///
    /// # Panics
    ///
    /// - Panics if unable to retrieve current working directory when opening a file.
    pub fn layout(
        &mut self,
        egui_context: egui::Context,
        window_size: [f32; 2],
        scale_factor: f64,
    ) {
        let GuiState {
            scene_texture_id, ..
        } = self;

        // We all love dark mode :D
        egui_context.set_visuals(egui::Visuals::dark());

        // Create a central frame for the GUI.
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

        // Main menu at the top.
        egui::TopBottomPanel::new(TopBottomSide::Top, Id::new("Main Menu")).show(
            &egui_context,
            |ui| {
                egui::menu::bar(ui, |ui| {
                    // File menu.
                    ui.menu_button("File", |ui| {
                        // Open.
                        if ui.button("Open file...").clicked() {
                            let current_dir =
                                std::env::current_dir().expect("Unable to get current directory");

                            let fd = rfd::FileDialog::new()
                                .set_directory(current_dir)
                                .add_filter("Wavefront (.obj)", &["obj"]);

                            if let Some(path) = fd.pick_file() {
                                self.file_path = path.display().to_string();
                            }

                            ui.close_menu();
                        }
                    });
                });
            },
        );

        // Status bar at the bottom.
        egui::TopBottomPanel::new(TopBottomSide::Bottom, Id::new("Status")).show(
            &egui_context,
            |ui| {
                // Display current file path.
                ui.horizontal(|ui| {
                    ui.label("File:");
                    ui.monospace(&self.file_path);
                });
            },
        );
    }
}

/// Registers a us image view for rendering the GUI.
fn get_scene_texture_id(gui: &mut Gui, scene_image: Arc<ImageView>) -> egui::TextureId {
    gui.register_user_image_view(scene_image, Default::default())
}
