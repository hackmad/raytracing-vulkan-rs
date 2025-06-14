use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
};

use anyhow::{Context, Result};
use glam::Vec3;
use log::info;
use serde::{Deserialize, Serialize};

use crate::{
    Camera, MAT_PROP_VALUE_TYPE_RGB, MAT_PROP_VALUE_TYPE_TEXTURE, MaterialColours, Mesh,
    PerspectiveCamera, RgbColour, shaders::closest_hit, texture::Textures,
};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CameraType {
    Perspective {
        name: String,
        eye: [f32; 3],
        look_at: [f32; 3],
        up: [f32; 3],
        fov_y: f32, // Vertical FOV in degrees.
        z_near: f32,
        z_far: f32,
        focal_length: f32,
        aperture_size: f32,
    },
}

impl CameraType {
    pub fn get_name(&self) -> &str {
        match self {
            Self::Perspective { name, .. } => name,
        }
    }

    pub fn to_camera(&self, image_width: u32, image_height: u32) -> Arc<RwLock<dyn Camera>> {
        match self {
            CameraType::Perspective {
                name: _,
                eye,
                look_at,
                up,
                fov_y,
                z_near,
                z_far,
                focal_length,
                aperture_size,
            } => Arc::new(RwLock::new(PerspectiveCamera::new(
                Vec3::from_slice(eye),
                Vec3::from_slice(look_at),
                Vec3::from_slice(up),
                fov_y.to_radians(),
                *z_near,
                *z_far,
                *focal_length,
                *aperture_size,
                image_width,
                image_height,
            ))),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MaterialPropertyValue {
    Rgb([f32; 3]),
    TextureFile(String),
}

impl MaterialPropertyValue {
    fn get_material_colour(&self) -> Option<RgbColour> {
        match self {
            MaterialPropertyValue::Rgb(colour) => Some(RgbColour::from(colour)),
            _ => None,
        }
    }

    fn get_texture_path(&self) -> Option<String> {
        match self {
            MaterialPropertyValue::TextureFile(path) => Some(path.clone()),
            _ => None,
        }
    }

    pub fn to_shader(
        &self,
        textures: &Textures,
        material_colours: &MaterialColours,
    ) -> closest_hit::MaterialPropertyValue {
        match self {
            MaterialPropertyValue::Rgb(colour) => material_colours
                .indices
                .get(&colour.into())
                .map(|index| closest_hit::MaterialPropertyValue {
                    propValueType: MAT_PROP_VALUE_TYPE_RGB,
                    index: *index,
                })
                .unwrap(),

            MaterialPropertyValue::TextureFile(path) => textures
                .indices
                .get(path)
                .map(|index| closest_hit::MaterialPropertyValue {
                    propValueType: MAT_PROP_VALUE_TYPE_TEXTURE,
                    index: *index,
                })
                .unwrap(),
        }
    }

    fn adjust_relative_path(&mut self, relative_to: &Path) {
        if let MaterialPropertyValue::TextureFile(path) = self {
            let path_buf = Path::new(path).to_path_buf();
            if path_buf.is_relative() {
                let mut new_path_buf = relative_to.to_path_buf();
                new_path_buf.push(path_buf);
                *path = new_path_buf.to_str().unwrap().to_owned();
            }
        }
    }
}

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

    fn get_texture_paths(&self) -> Vec<String> {
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

    fn adjust_relative_path(&mut self, relative_to: &Path) {
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

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectType {
    UvSphere {
        name: String,
        center: [f32; 3],
        radius: f32,
        rings: u32,
        segments: u32,
        material: String,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Render {
    pub camera: String,
    pub samples_per_pixel: u32, // See ray_gen.glsl. Don't exceed 64.
    pub sample_batches: u32,    // See ray_gen.glsl. Don't exceed 32.
    pub max_ray_depth: u32,
}

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
