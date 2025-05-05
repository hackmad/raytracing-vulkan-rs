use super::{MaterialPropertyDataEnum, shaders::closest_hit};
use anyhow::{Context, Result};
use std::{collections::HashSet, path::PathBuf, sync::Arc};
use vulkano::{
    DeviceSize,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryCommandBufferAbstract,
        allocator::CommandBufferAllocator,
    },
    device::Queue,
    memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter},
    sync::GpuFuture,
};

#[derive(Debug)]
pub struct ModelMaterial {
    pub diffuse: MaterialPropertyDataEnum,
}

pub struct Model {
    vertices: Vec<closest_hit::MeshVertex>,
    indices: Vec<u32>,
    pub material: Option<ModelMaterial>,
}

impl Model {
    pub fn load_obj(path: &str) -> Result<Vec<Self>> {
        let (models, materials) = tobj::load_obj(path, &tobj::GPU_LOAD_OPTIONS)?;

        let materials = &materials?;

        let parent_path = PathBuf::from(path)
            .parent()
            .context(format!("Invalid path {path}"))?
            .to_path_buf();

        let models: Vec<Self> = models
            .iter()
            .map(|model| {
                let mut vertices = vec![];
                let mut indices = vec![];

                let mesh = &model.mesh;

                for index in mesh.indices.iter() {
                    let pos_offset = (3 * index) as usize;
                    let tex_coord_offset = (2 * index) as usize;

                    let vertex = closest_hit::MeshVertex {
                        position: [
                            mesh.positions[pos_offset],
                            mesh.positions[pos_offset + 1],
                            mesh.positions[pos_offset + 2],
                        ],
                        normal: [
                            mesh.normals[pos_offset],
                            mesh.normals[pos_offset + 1],
                            mesh.normals[pos_offset + 2],
                        ],
                        tex_coord: [
                            mesh.texcoords[tex_coord_offset],
                            1.0 - mesh.texcoords[tex_coord_offset + 1],
                        ],
                    };

                    let vertex_index = vertices.len() as u32;

                    vertices.push(vertex);
                    indices.push(vertex_index);
                }

                /*
                println!(
                    "Vertex count: {}, Indices count: {}",
                    vertices.len(),
                    indices.len()
                );

                for (i, v) in vertices.iter().enumerate() {
                    println!(
                        "{i} {{position: {:?}, normal: {:?}, tex_coord: {:?}}}",
                        v.position, v.normal, v.tex_coord,
                    );
                }
                println!("{indices:?}");
                */

                let material = mesh.material_id.map(|mat_id| {
                    let mat = &materials[mat_id];
                    let diffuse = get_material_property(
                        &mat.diffuse,
                        &mat.diffuse_texture,
                        parent_path.clone(),
                    );
                    ModelMaterial { diffuse }
                });

                Self {
                    vertices,
                    indices,
                    material,
                }
            })
            .collect();

        Ok(models)
    }

    /// Create a vertex buffer for buildng the acceleration structure.
    pub fn create_blas_vertex_buffer(
        &self,
        memory_allocator: Arc<dyn MemoryAllocator>,
        command_buffer_allocator: Arc<dyn CommandBufferAllocator>,
        queue: Arc<Queue>,
    ) -> Result<Subbuffer<[closest_hit::MeshVertex]>> {
        create_device_local_buffer(
            memory_allocator,
            command_buffer_allocator,
            queue,
            BufferUsage::VERTEX_BUFFER
                | BufferUsage::SHADER_DEVICE_ADDRESS
                | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
            self.vertices.clone(),
        )
    }

    /// Create an index buffer for buildng the acceleration structure.
    pub fn create_blas_index_buffer(
        &self,
        memory_allocator: Arc<dyn MemoryAllocator>,
        command_buffer_allocator: Arc<dyn CommandBufferAllocator>,
        queue: Arc<Queue>,
    ) -> Result<Subbuffer<[u32]>> {
        create_device_local_buffer(
            memory_allocator,
            command_buffer_allocator,
            queue,
            BufferUsage::INDEX_BUFFER
                | BufferUsage::SHADER_DEVICE_ADDRESS
                | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
            self.indices.clone(),
        )
    }

    /// Create a storage buffer for accessing vertices in shader code.
    pub fn create_vertices_storage_buffer(
        &self,
        memory_allocator: Arc<dyn MemoryAllocator>,
        command_buffer_allocator: Arc<dyn CommandBufferAllocator>,
        queue: Arc<Queue>,
    ) -> Result<Subbuffer<[closest_hit::MeshVertex]>> {
        create_device_local_buffer(
            memory_allocator,
            command_buffer_allocator,
            queue,
            BufferUsage::STORAGE_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
            self.vertices.clone(),
        )
    }

    /// Create a storage buffer for accessing indices in shader code.
    pub fn create_indices_storage_buffer(
        &self,
        memory_allocator: Arc<dyn MemoryAllocator>,
        command_buffer_allocator: Arc<dyn CommandBufferAllocator>,
        queue: Arc<Queue>,
    ) -> Result<Subbuffer<[u32]>> {
        create_device_local_buffer(
            memory_allocator,
            command_buffer_allocator,
            queue,
            BufferUsage::STORAGE_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
            self.indices.clone(),
        )
    }

    /// Return a set of all texture paths.
    pub fn get_texture_paths(&self) -> HashSet<String> {
        let mut paths = HashSet::new();

        if let Some(mat) = &self.material {
            if let MaterialPropertyDataEnum::Texture { path } = &mat.diffuse {
                paths.insert(path.clone());
            }
        }

        paths
    }
}

fn get_material_property(
    color: &Option<[f32; 3]>,
    texture: &Option<String>,
    mut parent_path: PathBuf,
) -> MaterialPropertyDataEnum {
    match color {
        Some(c) => MaterialPropertyDataEnum::RGB { color: c.clone() },

        None => texture
            .clone()
            .map_or(MaterialPropertyDataEnum::None, |path| {
                if PathBuf::from(&path).is_absolute() {
                    MaterialPropertyDataEnum::Texture { path }
                } else {
                    parent_path.push(&path);

                    if let Some(path) = parent_path.to_str() {
                        MaterialPropertyDataEnum::Texture {
                            path: path.to_string(),
                        }
                    } else {
                        println!("Invalid texture path {path}.");
                        MaterialPropertyDataEnum::None
                    }
                }
            }),
    }
}

/// This will create buffers that can be accessed only by the GPU. One specific use case is to
/// access them via device addresses in shaders.
pub fn create_device_local_buffer<T, I>(
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: Arc<dyn CommandBufferAllocator>,
    queue: Arc<Queue>,
    usage: BufferUsage,
    data: I,
) -> Result<Subbuffer<[T]>>
where
    T: BufferContents,
    I: IntoIterator<Item = T>,
    I::IntoIter: ExactSizeIterator,
{
    let iter = data.into_iter();
    let size = iter.len() as DeviceSize;

    let temporary_accessible_buffer = Buffer::from_iter(
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
        iter,
    )?;

    let device_local_buffer = Buffer::new_slice::<T>(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: usage | BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        size,
    )?;

    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )?;

    builder.copy_buffer(CopyBufferInfo::buffers(
        temporary_accessible_buffer,
        device_local_buffer.clone(),
    ))?;

    builder
        .build()?
        .execute(queue.clone())?
        .then_signal_fence_and_flush()?
        .wait(None /* timeout */)?;

    Ok(device_local_buffer)
}
