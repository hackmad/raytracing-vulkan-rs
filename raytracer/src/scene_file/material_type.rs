use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{scene_file::material_property_value::MaterialPropertyValue, textures::RgbColour};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MaterialType {
    Lambertian {
        name: String,
        albedo: MaterialPropertyValue,
    },
    Metal {
        name: String,
        albedo: MaterialPropertyValue,
        fuzz: MaterialPropertyValue,
    },
    Dielectric {
        name: String,
        refraction_index: f32,
    },
}

impl MaterialType {
    pub fn get_name(&self) -> &str {
        match self {
            Self::Lambertian { name, .. } => name.as_ref(),
            Self::Metal { name, .. } => name.as_ref(),
            Self::Dielectric { name, .. } => name.as_ref(),
        }
    }

    pub fn get_material_colours(&self) -> Vec<RgbColour> {
        let rgb_colours = match self {
            Self::Lambertian { albedo, .. } => vec![albedo.get_material_colour()],
            Self::Metal { albedo, fuzz, .. } => {
                vec![albedo.get_material_colour(), fuzz.get_material_colour()]
            }
            Self::Dielectric { .. } => vec![],
        };
        rgb_colours.into_iter().flatten().collect()
    }

    pub fn get_texture_paths(&self) -> Vec<String> {
        let paths = match self {
            Self::Lambertian { albedo, .. } => {
                vec![albedo.get_texture_path()]
            }
            Self::Metal { albedo, fuzz, .. } => {
                vec![albedo.get_texture_path(), fuzz.get_texture_path()]
            }
            Self::Dielectric { .. } => vec![],
        };
        paths.into_iter().flatten().collect()
    }

    pub fn adjust_relative_path(&mut self, relative_to: &Path) {
        match self {
            Self::Lambertian { albedo, .. } => {
                albedo.adjust_relative_path(relative_to);
            }
            Self::Metal { albedo, fuzz, .. } => {
                albedo.adjust_relative_path(relative_to);
                fuzz.adjust_relative_path(relative_to);
            }
            Self::Dielectric { .. } => {}
        }
    }
}
