use std::{
    collections::{HashMap, hash_map::Entry},
    fmt,
    sync::Arc,
};

use anyhow::Result;
use image::{GenericImageView, ImageReader};
use log::info;
use vulkano::{
    DeviceSize,
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo,
        PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract,
    },
    format::Format,
    image::{Image, ImageCreateInfo, ImageType, ImageUsage, view::ImageView},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
};

use crate::{MAT_PROP_VALUE_TYPE_IMAGE, TextureType, Vk, shaders::closest_hit};

/// Stores texture image views that will be added to a `SampledImage` variable descriptor used by
/// the shader.
pub struct ImageTextures {
    /// The texture image views used by the shaders.
    pub image_views: Vec<Arc<ImageView>>,

    /// Maps unique texture paths to their index in `image_view`. These indices are used in the
    /// MaterialPropertyValue structure.
    pub indices: HashMap<String, u32>,
}

impl fmt::Debug for ImageTextures {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImageTextures")
            .field("image_views", &self.image_views.len())
            .field("indices", &self.indices)
            .finish()
    }
}

impl ImageTextures {
    /// Load all unique texture paths from all scene objects. Assumes images have alpha channel.
    pub fn load(vk: Arc<Vk>, textures: &HashMap<String, TextureType>) -> Result<Self> {
        let mut image_views = vec![];
        let mut indices = HashMap::new();

        let mut builder = AutoCommandBufferBuilder::primary(
            vk.command_buffer_allocator.clone(),
            vk.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        for texture in textures.values() {
            match texture {
                TextureType::Image { name, path } => {
                    if let Entry::Vacant(e) = indices.entry(name.clone()) {
                        let texture = load_texture(vk.clone(), path, &mut builder)?;
                        e.insert(image_views.len() as u32);
                        image_views.push(texture);
                    }
                }
                TextureType::Constant { .. } => {}
                TextureType::Checker { .. } => {}
            }
        }

        let _ = builder.build()?.execute(vk.queue.clone())?;

        Ok(Self {
            image_views,
            indices,
        })
    }

    pub fn to_shader(&self, name: &str) -> Option<closest_hit::MaterialPropertyValue> {
        self.indices
            .get(name)
            .map(|i| closest_hit::MaterialPropertyValue {
                propValueType: MAT_PROP_VALUE_TYPE_IMAGE,
                index: *i,
            })
    }
}

/// Loads the image texture into an new image view. Assumes image has alpha.
fn load_texture(
    vk: Arc<Vk>,
    path: &str,
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
) -> Result<Arc<ImageView>> {
    info!("Loading texture {path}...");

    let img = ImageReader::open(path)?.with_guessed_format()?.decode()?;
    let (width, height) = img.dimensions();

    info!("Loaded texture {path}: {width} x {height}");

    let image = Image::new(
        vk.memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R8G8B8A8_SRGB, // Needs to match image format from device.
            extent: [width, height, 1],
            array_layers: 1,
            usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    )?;

    let buffer: Subbuffer<[u8]> = Buffer::new_slice(
        vk.memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        (width * height * 4) as DeviceSize, // RGBA = 4
    )?;

    {
        let mut writer = buffer.write()?;
        writer.copy_from_slice(img.as_bytes());
    }

    builder.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(buffer, image.clone()))?;

    let image_view = ImageView::new_default(image)?;

    Ok(image_view)
}
