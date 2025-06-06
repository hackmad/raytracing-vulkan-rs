use crate::raytracer::{MaterialPropertyData, MaterialPropertyType};

use super::{MaterialColours, MaterialPropertyValue, Vk, shaders::closest_hit, texture::Textures};
use anyhow::{Context, Result, anyhow};
use log::debug;
use std::{collections::HashSet, path::PathBuf, sync::Arc};
use vulkano::{
    DeviceSize,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryCommandBufferAbstract,
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    sync::GpuFuture,
};

/// Materials for a given `Model`.
#[derive(Debug)]
pub struct ModelMaterial {
    /// Diffuse property.
    pub diffuse: MaterialPropertyValue,
}

/// The model.
pub struct Model {
    /// Vertex data for the mesh.
    vertices: Vec<closest_hit::MeshVertex>,

    /// Vertex indices for the mesh.
    indices: Vec<u32>,

    /// Material.
    pub material: Option<ModelMaterial>,
}

impl Model {
    /// Load a Wavefront OBJ file.
    pub fn load_obj(path: &str) -> Result<Vec<Self>> {
        let (models, materials) = tobj::load_obj(path, &tobj::GPU_LOAD_OPTIONS)?;

        let materials = &materials?;

        let parent_path = PathBuf::from(path)
            .parent()
            .context(format!("Invalid path {path}"))?
            .to_path_buf();

        let models: Vec<Self> = models
            .iter()
            .enumerate()
            .map(|(i, model)| {
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
                        texCoord: [
                            mesh.texcoords[tex_coord_offset],
                            1.0 - mesh.texcoords[tex_coord_offset + 1],
                        ],
                    };

                    let vertex_index = vertices.len() as u32;

                    vertices.push(vertex);
                    indices.push(vertex_index);
                }

                debug!(
                    "Vertex count: {}, Indices count: {}",
                    vertices.len(),
                    indices.len()
                );

                for (i, v) in vertices.iter().enumerate() {
                    debug!(
                        "{i} {{position: {:?}, normal: {:?}, tex_coord: {:?}}}",
                        v.position, v.normal, v.texCoord,
                    );
                }
                debug!("{indices:?}");

                let material = mesh.material_id.map(|mat_id| {
                    let mat = &materials[mat_id];
                    let diffuse = MaterialPropertyValue::new(
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
        vk: Arc<Vk>,
    ) -> Result<Subbuffer<[closest_hit::MeshVertex]>> {
        create_device_local_buffer(
            vk.clone(),
            BufferUsage::VERTEX_BUFFER
                | BufferUsage::SHADER_DEVICE_ADDRESS
                | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
            self.vertices.clone(),
        )
    }

    /// Create an index buffer for buildng the acceleration structure.
    pub fn create_blas_index_buffer(&self, vk: Arc<Vk>) -> Result<Subbuffer<[u32]>> {
        create_device_local_buffer(
            vk.clone(),
            BufferUsage::INDEX_BUFFER
                | BufferUsage::SHADER_DEVICE_ADDRESS
                | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
            self.indices.clone(),
        )
    }

    /// Create a storage buffer for accessing vertices in shader code.
    pub fn create_vertices_storage_buffer(
        &self,
        vk: Arc<Vk>,
    ) -> Result<Subbuffer<[closest_hit::MeshVertex]>> {
        create_device_local_buffer(
            vk.clone(),
            BufferUsage::STORAGE_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
            self.vertices.clone(),
        )
    }

    /// Create a storage buffer for accessing indices in shader code.
    pub fn create_indices_storage_buffer(&self, vk: Arc<Vk>) -> Result<Subbuffer<[u32]>> {
        create_device_local_buffer(
            vk.clone(),
            BufferUsage::STORAGE_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
            self.indices.clone(),
        )
    }

    /// Create a storage buffer for accessing materials in shader code.
    pub fn create_material_storage_buffer(
        &self,
        vk: Arc<Vk>,
        textures: &Textures,
        material_colours: &MaterialColours,
    ) -> Result<Subbuffer<[closest_hit::Material]>> {
        let diffuse = if let Some(material) = &self.material {
            MaterialPropertyData::from_property_value(
                MaterialPropertyType::Diffuse,
                &material.diffuse,
                &textures.indices,
                &material_colours.indices,
            )
        } else {
            MaterialPropertyData::new_none(MaterialPropertyType::Diffuse)
        };
        debug!("{diffuse:?}");

        let materials = vec![diffuse.into()]; // Order should respect `MAT_PROP_TYPE_*` indices

        create_device_local_buffer(
            vk.clone(),
            BufferUsage::STORAGE_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
            materials,
        )
    }

    /// Return a set of all texture paths.
    pub fn get_texture_paths(&self) -> HashSet<String> {
        let mut paths = HashSet::new();

        if let Some(mat) = &self.material {
            if let MaterialPropertyValue::Texture { path } = &mat.diffuse {
                paths.insert(path.clone());
            }
        }

        paths
    }
}

/// This will create buffers that can be accessed only by the GPU. One specific use case is to
/// access them via device addresses in shaders.
pub fn create_device_local_buffer<T, I>(
    vk: Arc<Vk>,
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

    if size == 0 {
        return Err(anyhow!("Cannot create device local buffer with empty data"));
    }

    let temporary_accessible_buffer = Buffer::from_iter(
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
        iter,
    )?;

    let device_local_buffer = Buffer::new_slice::<T>(
        vk.memory_allocator.clone(),
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
        vk.command_buffer_allocator.clone(),
        vk.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )?;

    builder.copy_buffer(CopyBufferInfo::buffers(
        temporary_accessible_buffer,
        device_local_buffer.clone(),
    ))?;

    builder
        .build()?
        .execute(vk.queue.clone())?
        .then_signal_fence_and_flush()?
        .wait(None /* timeout */)?;

    Ok(device_local_buffer)
}

/// This will create 2 storage buffers that can be accessed by their device address only by the GPU for the vertices
/// and indices. These addresses will be packed in another storage buffer representing the mesh data which will be
/// returned.
pub fn create_mesh_storage_buffer(
    vk: Arc<Vk>,
    models: &[Model],
    textures: &Textures,
    material_colours: &MaterialColours,
) -> Result<Subbuffer<[closest_hit::Mesh]>> {
    let vertices_storage_buffers = models
        .iter()
        .map(|model| model.create_vertices_storage_buffer(vk.clone()))
        .collect::<Result<Vec<_>>>()?;

    let indices_storage_buffers = models
        .iter()
        .map(|model| model.create_indices_storage_buffer(vk.clone()))
        .collect::<Result<Vec<_>>>()?;

    let materials_storage_buffers = models
        .iter()
        .map(|model| model.create_material_storage_buffer(vk.clone(), textures, material_colours))
        .collect::<Result<Vec<_>>>()?;

    let vertices_buffer_device_addresses = vertices_storage_buffers
        .iter()
        .map(|buf| {
            buf.device_address()
                .map(|addr| addr.get())
                .map_err(|e| anyhow!(e.to_string()))
        })
        .collect::<Result<Vec<_>>>()?;

    let indices_buffer_device_addresses = indices_storage_buffers
        .iter()
        .map(|buf| {
            buf.device_address()
                .map(|addr| addr.get())
                .map_err(|e| anyhow!(e.to_string()))
        })
        .collect::<Result<Vec<_>>>()?;

    let materials_buffer_device_addresses = materials_storage_buffers
        .iter()
        .map(|buf| {
            buf.device_address()
                .map(|addr| addr.get())
                .map_err(|e| anyhow!(e.to_string()))
        })
        .collect::<Result<Vec<_>>>()?;

    let meshes = vertices_buffer_device_addresses
        .into_iter()
        .zip(indices_buffer_device_addresses)
        .zip(materials_buffer_device_addresses)
        .map(
            |((vertices_ref, indices_ref), materials_ref)| closest_hit::Mesh {
                verticesRef: vertices_ref,
                indicesRef: indices_ref,
                materialsRef: materials_ref,
            },
        );

    let data = Buffer::from_iter(
        vk.memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        meshes,
    )?;

    Ok(data)
}
