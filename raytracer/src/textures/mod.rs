mod checker_texture;
mod constant_colour_texture;
mod image_texture;
mod noise_texture;

use std::sync::Arc;

use anyhow::Result;
pub use checker_texture::*;
pub use constant_colour_texture::*;
pub use image_texture::*;
use log::debug;
pub use noise_texture::*;
use vulkano::buffer::{BufferUsage, Subbuffer};

use crate::{
    MAT_PROP_VALUE_TYPE_RGB, SceneFile, Vk, create_device_local_buffer, shaders::closest_hit,
};

pub struct Textures {
    pub constant_colour_textures: ConstantColourTextures,
    pub image_textures: ImageTextures,
    pub checker_textures: CheckerTextures,
    pub noise_textures: NoiseTextures,
}

impl Textures {
    pub fn new(vk: Arc<Vk>, scene_file: &SceneFile) -> Result<Self> {
        let all_textures = scene_file.get_textures();

        for texture in scene_file.textures.iter() {
            texture.is_valid(&all_textures)?;
        }

        let constant_colour_textures = ConstantColourTextures::new(&all_textures);
        let image_textures = ImageTextures::load(vk, &all_textures)?;
        let checker_textures = CheckerTextures::new(&all_textures);
        let noise_textures = NoiseTextures::new(&all_textures);

        debug!("{constant_colour_textures:?}");
        debug!("{image_textures:?}");

        Ok(Self {
            constant_colour_textures,
            image_textures,
            checker_textures,
            noise_textures,
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
        if let Some(v) = self.checker_textures.to_shader(name) {
            return Some(v);
        }
        if let Some(v) = self.noise_textures.to_shader(name) {
            return Some(v);
        }
        None
    }

    /// Create a storage buffers for accessing materials in shader code.
    pub fn create_buffers(&self, vk: Arc<Vk>) -> Result<TextureBuffers> {
        let buffer_usage = BufferUsage::STORAGE_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS;

        // Note: We can't create buffers from empty list. So use a texture and push constants
        // will set the number of textures to 0 which the shader code checks for out of bounds.

        debug!("Creating checker texture storage buffer");
        let checker_buffer = create_device_local_buffer(
            vk.clone(),
            buffer_usage,
            if !self.checker_textures.textures.is_empty() {
                self.checker_textures
                    .textures
                    .iter()
                    .map(|t| closest_hit::CheckerTexture {
                        scale: t.scale,
                        odd: self.to_shader(&t.odd).unwrap(), // TODO could return Err() when odd/even not found.
                        even: self.to_shader(&t.even).unwrap(),
                    })
                    .collect()
            } else {
                vec![closest_hit::CheckerTexture {
                    scale: 1.0,
                    odd: closest_hit::MaterialPropertyValue {
                        propValueType: MAT_PROP_VALUE_TYPE_RGB,
                        index: 0,
                    },
                    even: closest_hit::MaterialPropertyValue {
                        propValueType: MAT_PROP_VALUE_TYPE_RGB,
                        index: 0,
                    },
                }]
            },
        )?;

        debug!("Creating noise texture storage buffer");
        let noise_buffer = create_device_local_buffer(
            vk.clone(),
            buffer_usage,
            if !self.noise_textures.textures.is_empty() {
                self.noise_textures
                    .textures
                    .iter()
                    .map(|t| closest_hit::NoiseTexture { scale: t.scale })
                    .collect()
            } else {
                vec![closest_hit::NoiseTexture { scale: 1.0 }]
            },
        )?;

        Ok(TextureBuffers {
            checker: checker_buffer,
            noise: noise_buffer,
        })
    }
}

/// Holds the storage buffers for the textures other than constant colour and image types.
pub struct TextureBuffers {
    pub checker: Subbuffer<[closest_hit::CheckerTexture]>,
    pub noise: Subbuffer<[closest_hit::NoiseTexture]>,
}
