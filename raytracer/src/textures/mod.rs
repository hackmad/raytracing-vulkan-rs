mod constant_colour_texture;
mod image_texture;

use std::sync::Arc;

use anyhow::Result;
pub use constant_colour_texture::*;
pub use image_texture::*;
use log::debug;

use crate::{SceneFile, Vk, shaders::closest_hit};

pub struct Textures {
    pub constant_colour_textures: ConstantColourTextures,
    pub image_textures: ImageTextures,
}

impl Textures {
    pub fn new(vk: Arc<Vk>, scene_file: &SceneFile) -> Result<Self> {
        let textures = scene_file.get_textures();
        let constant_colour_textures = ConstantColourTextures::new(&textures);
        let image_textures = ImageTextures::load(vk, &textures)?;

        debug!("{constant_colour_textures:?}");
        debug!("{image_textures:?}");

        Ok(Self {
            constant_colour_textures,
            image_textures,
        })
    }

    pub fn to_shader(&self, name: &str) -> Option<closest_hit::MaterialPropertyValue> {
        // Texture names will be unique across all texture types.
        if let Some(v) = self.constant_colour_textures.to_shader(name) {
            return Some(v);
        }
        if let Some(v) = self.image_textures.to_shader(name) {
            return Some(v);
        }
        None
    }
}
