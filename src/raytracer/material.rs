use std::collections::HashMap;

use vulkano::buffer::BufferContents;

const MAT_PROP_NONE: u8 = 0;
const MAT_PROP_RGB: u8 = 1;
const MAT_PROP_TEXTURE: u8 = 2;

#[repr(C)]
#[derive(BufferContents, Clone, Copy)]
pub struct MaterialPropertyData {
    pub prop_type: u8,
    pub color: [f32; 3],
    pub texture_index: i32,
}

impl MaterialPropertyData {
    pub fn new_color(rgb: &[f32; 3]) -> Self {
        Self {
            prop_type: MAT_PROP_RGB,
            color: rgb.clone(),
            texture_index: -1,
        }
    }

    pub fn new_texture(index: i32) -> Self {
        Self {
            prop_type: MAT_PROP_TEXTURE,
            color: [0.0, 0.0, 0.0],
            texture_index: index,
        }
    }
}

impl Default for MaterialPropertyData {
    fn default() -> Self {
        Self {
            prop_type: MAT_PROP_NONE,
            color: [0.0, 0.0, 0.0],
            texture_index: -1,
        }
    }
}

#[derive(Debug, Eq, Hash, PartialEq)]
pub enum MaterialProperty {
    Diffuse,
}

#[derive(Clone, Debug)]
pub enum MaterialPropertyDataEnum {
    None,
    RGB { color: [f32; 3] },
    Texture { path: String },
}

pub fn get_material_data(
    prop_type: &MaterialPropertyDataEnum,
    texture_indices: &HashMap<String, i32>,
) -> MaterialPropertyData {
    match prop_type {
        MaterialPropertyDataEnum::None => MaterialPropertyData::default(),
        MaterialPropertyDataEnum::RGB { color } => MaterialPropertyData::new_color(color),
        MaterialPropertyDataEnum::Texture { path } => {
            let texture_index = texture_indices
                .get(path)
                .expect(format!("Texture {path} not found").as_ref());
            MaterialPropertyData::new_texture(*texture_index)
        }
    }
}
