use anyhow::{Context, Result, anyhow};
use std::{collections::HashSet, path::PathBuf, sync::Arc};

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter},
};

use super::{MaterialPropertyDataEnum, geometry::VertexData};

#[derive(Debug)]
pub struct ModelMaterial {
    pub diffuse: MaterialPropertyDataEnum,
}

pub struct Model {
    vertices: Vec<VertexData>,
    indices: Vec<u32>,
    pub material: Option<ModelMaterial>,
}

impl Model {
    pub fn cube() -> Self {
        #[rustfmt::skip]
        let vertices = vec![
        ];

        #[rustfmt::skip]
        let indices = vec![
        ];

        let material = Some(ModelMaterial {
            diffuse: MaterialPropertyDataEnum::Texture {
                path: "assets/obj/test-grid.png".to_string(),
            },
        });

        Self {
            vertices,
            indices,
            material,
        }
    }

    pub fn triangle() -> Self {
        #[rustfmt::skip]
        let vertices = vec![
            VertexData { position: [-1.0,  1.0,  0.0], normal: [ 0.0,  0.0,  1.0], tex_coord: [0.0, 1.0] },
            VertexData { position: [ 1.0,  1.0,  0.0], normal: [ 0.0,  0.0,  1.0], tex_coord: [1.0, 1.0] },
            VertexData { position: [ 0.0, -1.0,  0.0], normal: [ 0.0,  0.0,  1.0], tex_coord: [0.5, 0.0] },
        ];

        #[rustfmt::skip]
        let indices = vec![0, 1, 2];

        let material = Some(ModelMaterial {
            diffuse: MaterialPropertyDataEnum::Texture {
                path: "assets/obj/test-grid.png".to_string(),
            },
        });

        Self {
            vertices,
            indices,
            material,
        }
    }

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
                            mesh.texcoords[tex_coord_offset + 1],
                        ],
                    };

                    let vertex_index = vertices.len() as u32;
                    vertices.push(vertex);
                    indices.push(vertex_index);
                }
                println!("Vertices: {}", vertices.len());
                println!("Indices:  {}", indices.len());

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
