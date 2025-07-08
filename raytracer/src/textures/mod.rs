mod checker_texture;
mod constant_colour_texture;
mod image_texture;
mod noise_texture;

use std::sync::Arc;

use anyhow::Result;
use ash::vk;
pub use checker_texture::*;
pub use constant_colour_texture::*;
pub use image_texture::*;
use log::debug;
pub use noise_texture::*;
use scene_file::SceneFile;
use shaders::MaterialPropertyValue;
use vulkan::{Buffer, VulkanContext};

pub struct Textures {
    pub constant_colour_textures: ConstantColourTextures,
    pub image_textures: ImageTextures,
    pub checker_textures: CheckerTextures,
    pub noise_textures: NoiseTextures,
}

impl Textures {
    pub fn new(context: Arc<VulkanContext>, scene_file: &SceneFile) -> Result<Self> {
        let all_textures = scene_file.get_textures();

        for texture in scene_file.textures.iter() {
            texture.is_valid(&all_textures)?;
        }

        let constant_colour_textures = ConstantColourTextures::new(&all_textures);
        let image_textures = ImageTextures::load(context, &all_textures)?;
        let checker_textures = CheckerTextures::new(&all_textures);
        let noise_textures = NoiseTextures::new(&all_textures);

        debug!("{constant_colour_textures:?}");
        debug!("{image_textures:?}");
        debug!("{checker_textures:?}");
        debug!("{noise_textures:?}");

        Ok(Self {
            constant_colour_textures,
            image_textures,
            checker_textures,
            noise_textures,
        })
    }

    pub fn to_shader(&self, name: &str) -> Option<MaterialPropertyValue> {
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
    pub fn create_buffers(&self, context: Arc<VulkanContext>) -> Result<TextureBuffers> {
        let buffer_usage =
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;

        // Note: We can't create buffers from empty list. So use a texture and push constants
        // will set the number of textures to 0 which the shader code checks for out of bounds.

        debug!("Creating checker texture storage buffer");
        let checker_textures: Vec<_> = self
            .checker_textures
            .textures
            .iter()
            .map(|t| shaders::CheckerTexture {
                scale: t.scale,
                odd: self.to_shader(&t.odd).unwrap(), // TODO could return Err() when odd/even not found.
                even: self.to_shader(&t.even).unwrap(),
            })
            .collect();

        let checker_buffer = Buffer::new_device_local_storage_buffer(
            context.clone(),
            buffer_usage,
            &checker_textures,
            "checker_buffer",
        )?;

        debug!("Creating noise texture storage buffer");
        let noise_textures: Vec<_> = self
            .noise_textures
            .textures
            .iter()
            .map(|t| shaders::NoiseTexture { scale: t.scale })
            .collect();

        let noise_buffer = Buffer::new_device_local_storage_buffer(
            context.clone(),
            buffer_usage,
            &noise_textures,
            "noise_buffer",
        )?;

        Ok(TextureBuffers {
            checker: checker_buffer,
            noise: noise_buffer,
        })
    }
}

/// Holds the storage buffers for the textures other than constant colour and image types.
pub struct TextureBuffers {
    pub checker: Buffer,
    pub noise: Buffer,
}
