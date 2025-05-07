use std::{collections::HashMap, path::PathBuf};

use super::shaders::closest_hit;

#[derive(Clone, Copy, Debug)]
#[repr(u32)]
pub enum MaterialPropertyType {
    Diffuse = 0,
}

#[derive(Clone, Copy, Debug)]
#[repr(u32)]
pub enum MaterialPropertyValueType {
    None = 0,
    RGB = 1,
    Texture = 2,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct MaterialPropertyData {
    pub prop_type: u32,
    pub prop_value_type: u32,
    pub color: [f32; 3],
    pub texture_index: i32,
}

impl MaterialPropertyData {
    pub fn new_color(prop_type: MaterialPropertyType, rgb: &[f32; 3]) -> Self {
        Self {
            prop_type: prop_type as _,
            prop_value_type: MaterialPropertyValueType::RGB as _,
            color: rgb.clone(),
            texture_index: -1,
        }
    }

    pub fn new_texture_index(prop_type: MaterialPropertyType, index: i32) -> Self {
        Self {
            prop_type: prop_type as _,
            prop_value_type: MaterialPropertyValueType::Texture as _,
            color: [0.0, 0.0, 0.0],
            texture_index: index,
        }
    }

    pub fn new_none(prop_type: MaterialPropertyType) -> Self {
        Self {
            prop_type: prop_type as _,
            prop_value_type: MaterialPropertyValueType::None as _,
            color: [0.0, 0.0, 0.0],
            texture_index: -1,
        }
    }

    pub fn from_property_value(
        prop_type: MaterialPropertyType,
        value: &MaterialPropertyValue,
        texture_indices: &HashMap<String, i32>,
    ) -> Self {
        match value {
            MaterialPropertyValue::None => Self::new_none(prop_type),
            MaterialPropertyValue::RGB { color } => Self::new_color(prop_type, color),
            MaterialPropertyValue::Texture { path } => {
                let texture_index = texture_indices
                    .get(path)
                    .expect(format!("Texture {path} not found").as_ref());
                Self::new_texture_index(prop_type, *texture_index)
            }
        }
    }
}

impl Into<closest_hit::Material> for MaterialPropertyData {
    fn into(self) -> closest_hit::Material {
        closest_hit::Material {
            propType: self.prop_type,
            propValueType: self.prop_value_type,
            color: self.color,
            textureIndex: self.texture_index,
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
