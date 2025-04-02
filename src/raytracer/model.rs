use anyhow::{Result, anyhow};
use std::{fs::File, io::BufReader, sync::Arc};

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter},
};

use super::geometry::VertexData;

pub struct Model {
    pub vertices: Vec<VertexData>,
    pub indices: Vec<u32>,
}

impl Model {
    pub fn load_obj(path: &str) -> Result<Vec<Self>> {
        let mut reader = BufReader::new(File::open(path)?);

        let (models, _materials) =
            tobj::load_obj_buf(&mut reader, &tobj::GPU_LOAD_OPTIONS, |_| {
                Ok(Default::default())
            })?;

        let models: Vec<Self> = models
            .iter()
            .map(|model| {
                let mut vertices = vec![];
                let mut indices = vec![];

                let mesh = &model.mesh;

                for index in mesh.indices.iter() {
                    let pos_offset = (3 * index) as usize;
                    let tex_coord_offset = (2 * index) as usize;

                    let vertex = VertexData {
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

                    vertices.push(vertex);
                    indices.push(indices.len() as u32);
                }

                Self { vertices, indices }
            })
            .collect();

        Ok(models)
    }

    pub fn create_vertex_buffer(
        &self,
        memory_allocator: Arc<dyn MemoryAllocator>,
    ) -> Result<Subbuffer<[VertexData]>> {
        Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER
                    | BufferUsage::SHADER_DEVICE_ADDRESS
                    | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            self.vertices.clone(),
        )
        .map_err(|e| anyhow!("{e:?}"))
    }

    pub fn create_index_buffer(
        &self,
        memory_allocator: Arc<dyn MemoryAllocator>,
    ) -> Result<Subbuffer<[u32]>> {
        Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER
                    | BufferUsage::SHADER_DEVICE_ADDRESS
                    | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            self.indices.clone(),
        )
        .map_err(|e| anyhow!("{e:?}"))
    }
}
