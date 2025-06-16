mod camera_type;
mod material_property_value;
mod material_type;
mod object_type;
mod render;

pub use camera_type::*;
pub use material_property_value::*;
pub use material_type::*;
pub use object_type::*;
pub use render::*;

use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use log::info;
use serde::{Deserialize, Serialize};

use crate::Mesh;

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct SceneFile {
    pub cameras: Vec<CameraType>,
    pub materials: Vec<MaterialType>,
    pub objects: Vec<ObjectType>,
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
        for material_type in self.materials.iter_mut() {
            material_type.adjust_relative_path(relative_to);
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

    /// Returns all unique texture paths from scene file.
    pub fn get_texture_paths(&self) -> HashSet<String> {
        let mut texture_paths = HashSet::new();

        for material_type in self.materials.iter() {
            for path in material_type.get_texture_paths() {
                texture_paths.insert(path);
            }
        }

        texture_paths
    }

    /// Return all meshes.
    pub fn get_meshes(&self) -> Vec<Mesh> {
        self.objects.iter().map(|o| o.into()).collect()
    }
}
