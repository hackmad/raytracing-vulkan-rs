use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{
    MAT_PROP_VALUE_TYPE_IMAGE, MAT_PROP_VALUE_TYPE_RGB,
    shaders::closest_hit,
    textures::{ConstantColourTextures, ImageTextures, RgbColour},
};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MaterialPropertyValue {
    Rgb([f32; 3]),
    TextureFile(String),
}

impl MaterialPropertyValue {
    pub fn get_material_colour(&self) -> Option<RgbColour> {
        match self {
            MaterialPropertyValue::Rgb(colour) => Some(RgbColour::from(colour)),
            _ => None,
        }
    }

    pub fn get_texture_path(&self) -> Option<String> {
        match self {
            MaterialPropertyValue::TextureFile(path) => Some(path.clone()),
            _ => None,
        }
    }

    pub fn to_shader(
        &self,
        image_textures: &ImageTextures,
        constant_colour_textures: &ConstantColourTextures,
    ) -> closest_hit::MaterialPropertyValue {
        match self {
            MaterialPropertyValue::Rgb(colour) => constant_colour_textures
                .indices
                .get(&colour.into())
                .map(|index| closest_hit::MaterialPropertyValue {
                    propValueType: MAT_PROP_VALUE_TYPE_RGB,
                    index: *index,
                })
                .unwrap(),

            MaterialPropertyValue::TextureFile(path) => image_textures
                .indices
                .get(path)
                .map(|index| closest_hit::MaterialPropertyValue {
                    propValueType: MAT_PROP_VALUE_TYPE_IMAGE,
                    index: *index,
                })
                .unwrap(),
        }
    }

    pub fn adjust_relative_path(&mut self, relative_to: &Path) {
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
