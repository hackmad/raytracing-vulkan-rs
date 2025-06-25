mod camera;
mod material;
mod primitive;
mod render;
mod sky;
mod texture;

pub use camera::*;
pub use material::*;
pub use primitive::*;
pub use render::*;
pub use sky::*;
pub use texture::*;

use std::{
    collections::{HashMap, hash_map::Entry},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use log::{info, warn};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct SceneFile {
    pub cameras: Vec<Camera>,
    pub textures: Vec<Texture>,
    pub materials: Vec<Material>,
    pub primitives: Vec<Primitive>,
    pub sky: Sky,
    pub render: Render,
}

impl SceneFile {
    pub fn load_json(path: &str) -> Result<Self> {
        let serialized = std::fs::read_to_string(path)?;
        let mut deserialized: Self = serde_json::from_str(&serialized)
            .with_context(|| format!("Unable to parse scene file '{path}'"))?;

        let path_buf = PathBuf::from(path);
        let relative_to = path_buf.parent().unwrap();
        deserialized.adjust_relative_paths(relative_to);
        deserialized.enforce_render_limits();

        Ok(deserialized)
    }

    pub fn save_json(&self, path: &str) -> Result<()> {
        let serialized = serde_json::to_string_pretty(self)?;
        std::fs::write(path, serialized)?;
        Ok(())
    }

    fn adjust_relative_paths(&mut self, relative_to: &Path) {
        for texture in self.textures.iter_mut() {
            texture.adjust_relative_path(relative_to);
        }
    }

    fn enforce_render_limits(&mut self) {
        if self.render.samples_per_pixel > 64 {
            info!(
                "Samples per pixel {} too high. Limiting to 64.",
                self.render.samples_per_pixel
            );
            self.render.samples_per_pixel = 64;
        }
        if self.render.sample_batches > 32 {
            info!(
                "Sample batches {} too high. Limiting to 32.",
                self.render.sample_batches
            );
            self.render.sample_batches = 32;
        }
    }

    // Note: Texture names will be unique across all texture types.
    pub fn get_textures(&self) -> HashMap<String, Texture> {
        let mut textures: HashMap<String, Texture> = HashMap::new();

        for texture in self.textures.iter() {
            let name = texture.get_name();
            if let Entry::Vacant(e) = textures.entry(name.to_string()) {
                e.insert(texture.clone());
            } else {
                warn!("Texture name '{name}' is used multiple times");
            }
        }

        textures
    }
}
