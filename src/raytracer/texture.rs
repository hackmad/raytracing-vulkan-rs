use anyhow::Result;
use image::{GenericImageView, ImageReader};
use std::{collections::HashMap, fmt, sync::Arc};
use vulkano::{
    DeviceSize,
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo,
        PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract, allocator::CommandBufferAllocator,
    },
    device::Queue,
    format::Format,
    image::{Image, ImageCreateInfo, ImageType, ImageUsage, view::ImageView},
    memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter},
};

use super::model::Model;

/// Stores texture image views.
pub struct Textures {
    /// The texture image views.
    pub image_views: Vec<Arc<ImageView>>,

    /// Maps unique texture paths to their index in `image_view`.
    pub indices: HashMap<String, i32>, /* GLSL int => i32*/
}
impl fmt::Debug for Textures {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Textures")
            .field("image_views", &self.image_views.len())
            .field("indices", &self.indices)
            .finish()
    }
}

impl Textures {
    /// Load all unique texture paths from all models. Assumes images have alpha channel.
    pub fn load(
        models: &[Model],
        memory_allocator: Arc<dyn MemoryAllocator>,
        command_buffer_allocator: Arc<dyn CommandBufferAllocator>,
        queue: Arc<Queue>,
    ) -> Result<Self> {
        let mut image_views = vec![];
        let mut indices: HashMap<String, i32> = HashMap::new();

        let mut builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        for model in models.iter() {
            for path in model.get_texture_paths() {
                if !indices.contains_key(&path) {
                    let texture = load_texture(&path, memory_allocator.clone(), &mut builder)?;
                    indices.insert(path.clone(), image_views.len() as i32);
                    image_views.push(texture);
                }
            }
        }

        let _ = builder.build()?.execute(queue.clone())?;

        Ok(Self {
            image_views,
            indices,
        })
    }
}

/// Loads the image texture into an new image view. Assumes image has alpha.
fn load_texture(
    path: &str,
    memory_allocator: Arc<dyn MemoryAllocator>,
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
) -> Result<Arc<ImageView>> {
    println!("Loading texture {path}...");

    let img = ImageReader::open(path)?.with_guessed_format()?.decode()?;
    let (width, height) = img.dimensions();

    println!("Loaded texture {path}: {width} x {height}");

    let image = Image::new(
        memory_allocator.clone(),
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
        memory_allocator.clone(),
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
