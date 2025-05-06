use std::{collections::HashMap, path::PathBuf};

use vulkano::buffer::BufferContents;

#[repr(C)]
#[derive(BufferContents, Clone, Copy)]
pub struct MaterialPropertyData {
    pub prop_type: u8,
    pub color: [f32; 3],
    pub texture_index: i32,
}

impl MaterialPropertyData {
    pub const MAT_PROP_NONE: u8 = 0;
    pub const MAT_PROP_RGB: u8 = 1;
    pub const MAT_PROP_TEXTURE: u8 = 2;

    pub fn new_color(rgb: &[f32; 3]) -> Self {
        Self {
            prop_type: Self::MAT_PROP_RGB,
            color: rgb.clone(),
            texture_index: -1,
        }
    }

    pub fn new_texture(index: i32) -> Self {
        Self {
            prop_type: Self::MAT_PROP_TEXTURE,
            color: [0.0, 0.0, 0.0],
            texture_index: index,
        }
    }

    pub fn from_property_type(
        prop_type: &MaterialPropertyValue,
        texture_indices: &HashMap<String, i32>,
    ) -> Self {
        match prop_type {
            MaterialPropertyValue::None => Self::default(),
            MaterialPropertyValue::RGB { color } => Self::new_color(color),
            MaterialPropertyValue::Texture { path } => {
                let texture_index = texture_indices
                    .get(path)
                    .expect(format!("Texture {path} not found").as_ref());
                Self::new_texture(*texture_index)
            }
        }
    }
}

impl Default for MaterialPropertyData {
    fn default() -> Self {
        Self {
            prop_type: Self::MAT_PROP_NONE,
            color: [0.0, 0.0, 0.0],
            texture_index: -1,
        }
    }
}

#[derive(Clone, Debug)]
pub enum MaterialPropertyValue {
    None,
    RGB { color: [f32; 3] },
    Texture { path: String },
}

impl MaterialPropertyValue {
    pub fn new(
        color: &Option<[f32; 3]>,
        texture: &Option<String>,
        mut parent_path: PathBuf,
    ) -> Self {
        match color {
            Some(c) => MaterialPropertyValue::RGB { color: c.clone() },

            None => texture.clone().map_or(Self::None, |path| {
                if PathBuf::from(&path).is_absolute() {
                    Self::Texture { path }
                } else {
                    parent_path.push(&path);

                    if let Some(path) = parent_path.to_str() {
                        Self::Texture {
                            path: path.to_string(),
                        }
                    } else {
                        println!("Invalid texture path {path}.");
                        Self::None
                    }
                }
            }),
        }
    }
}
