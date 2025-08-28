use std::{
    collections::{HashMap, hash_map::Entry},
    fmt,
    sync::Arc,
};

use anyhow::Result;
use image::{GenericImageView, ImageReader};
use log::info;
use scene_file::Texture;
use shaders::{MAT_PROP_VALUE_TYPE_IMAGE, MaterialPropertyValue};
use vulkan::{Image, VulkanContext};

/// Stores texture image views that will be added to a `SampledImage` variable descriptor used by
/// the shader.
pub struct ImageTextures {
    /// The texture image views used by the shaders.
    pub images: Vec<Image>,

    /// Maps unique texture paths to their index in `image_view`. These indices are used in the
    /// MaterialPropertyValue structure.
    pub indices: HashMap<String, u32>,
}

impl ImageTextures {
    /// Load all unique texture paths from all scene objects. Assumes images have alpha channel.
    pub fn load(context: Arc<VulkanContext>, textures: &HashMap<String, Texture>) -> Result<Self> {
        let mut images = vec![];
        let mut indices = HashMap::new();

        for texture in textures.values() {
            if let Texture::Image { name, path } = texture
                && let Entry::Vacant(e) = indices.entry(name.clone())
            {
                let texture = load_texture(context.clone(), path)?;
                e.insert(images.len() as u32);
                images.push(texture);
            }
        }

        Ok(Self { images, indices })
    }

    pub fn to_shader(&self, name: &str) -> Option<MaterialPropertyValue> {
        self.indices.get(name).map(|i| MaterialPropertyValue {
            prop_value_type: MAT_PROP_VALUE_TYPE_IMAGE,
            index: *i,
        })
    }
}

impl fmt::Debug for ImageTextures {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImageTextures")
            .field("images", &self.images.len())
            .field("indices", &self.indices)
            .finish()
    }
}

/// Loads the image texture into an new image view. Assumes image has alpha.
fn load_texture(context: Arc<VulkanContext>, path: &str) -> Result<Image> {
    info!("Loading texture {path}...");

    let img = ImageReader::open(path)?.with_guessed_format()?.decode()?;
    let (width, height) = img.dimensions();
    let colour_type = img.color();
    let channels = colour_type.channel_count();
    let rgba_image = img.to_rgba8();

    info!("Loaded texture {path}: {width} x {height} x {channels}");

    let image = Image::new_rgba_image(context.clone(), &rgba_image)?;
    Ok(image)
}
