use std::{f32::consts::PI, sync::Arc};

use anyhow::Result;
use glam::Vec3;
use log::{debug, warn};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
};

use crate::{
    MAT_TYPE_DIELECTRIC, MAT_TYPE_LAMBERTIAN, MAT_TYPE_METAL, MAT_TYPE_NONE, Materials, ObjectType,
    Vk, create_device_local_buffer, shaders::closest_hit,
};

// This is used for cleaner code and it represents the data that the shader's MeshVertex structure needs.
#[derive(Clone, Debug)]
pub struct Vertex {
    pub p: [f32; 3],
    pub n: [f32; 3],
    pub uv: [f32; 2],
}

impl Vertex {
    pub fn new(p: [f32; 3], n: [f32; 3], uv: [f32; 2]) -> Self {
        Self { p, n, uv }
    }
}

impl From<&Vertex> for closest_hit::MeshVertex {
    // Convert Vertex to shader struct.
    fn from(value: &Vertex) -> Self {
        Self {
            p: value.p,
            n: value.n,
            u: value.uv[0],
            v: value.uv[1],
        }
    }
}

#[derive(Debug)]
pub struct Mesh {
    pub name: String,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub material: String,
}

impl Mesh {
    /// Create a vertex buffer for buildng the acceleration structure.
    pub fn create_blas_vertex_buffer(
        &self,
        vk: Arc<Vk>,
    ) -> Result<Subbuffer<[closest_hit::MeshVertex]>> {
        debug!("Creating BLAS vertex buffer");
        create_device_local_buffer(
            vk.clone(),
            BufferUsage::VERTEX_BUFFER
                | BufferUsage::SHADER_DEVICE_ADDRESS
                | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
            self.vertices.iter().map(closest_hit::MeshVertex::from),
        )
    }

    /// Create an index buffer for buildng the acceleration structure.
    pub fn create_blas_index_buffer(&self, vk: Arc<Vk>) -> Result<Subbuffer<[u32]>> {
        debug!("Creating BLAS index buffer");
        create_device_local_buffer(
            vk.clone(),
            BufferUsage::INDEX_BUFFER
                | BufferUsage::SHADER_DEVICE_ADDRESS
                | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
            self.indices.clone(),
        )
    }
}

impl From<&ObjectType> for Mesh {
    fn from(value: &ObjectType) -> Self {
        match value {
            ObjectType::UvSphere {
                name,
                center,
                radius,
                rings,
                segments,
                material,
            } => {
                let (vertices, indices) = generate_uv_sphere(center, *radius, *rings, *segments);
                Mesh {
                    name: name.clone(),
                    vertices,
                    indices,
                    material: material.clone(),
                }
            }

            ObjectType::Triangle {
                name,
                points,
                normal,
                uv,
                material,
            } => {
                let vertices: Vec<_> = points
                    .iter()
                    .enumerate()
                    .map(|(i, p)| Vertex::new(*p, *normal, uv[i]))
                    .collect();
                Mesh {
                    name: name.clone(),
                    vertices,
                    indices: vec![0, 1, 2],
                    material: material.clone(),
                }
            }

            ObjectType::Quad {
                name,
                points,
                normal,
                uv,
                material,
            } => {
                let vertices: Vec<_> = points
                    .iter()
                    .enumerate()
                    .map(|(i, p)| Vertex::new(*p, *normal, uv[i]))
                    .collect();
                Mesh {
                    name: name.clone(),
                    vertices,
                    indices: vec![0, 1, 2, 0, 2, 3],
                    material: material.clone(),
                }
            }
        }
    }
}

fn uv_sphere_vertex(
    center: &Vec3,
    radius: f32,
    ring: u32,
    segment: u32,
    du: f32,
    dv: f32,
    top_or_bottom: bool,
) -> Vertex {
    let shift_u = if top_or_bottom { du / 2.0 } else { 0.0 };
    let u = segment as f32 * du + shift_u;
    let v = ring as f32 * dv;

    let theta = 2.0 * PI * u;
    let phi = PI * v;

    let n = Vec3::new(
        -phi.sin() * theta.cos(),
        -phi.cos(),
        phi.sin() * theta.sin(),
    );
    let p = center + radius * n;

    Vertex::new(p.into(), n.into(), [u, v])
}

fn generate_uv_sphere(
    center: &[f32; 3],
    radius: f32,
    rings: u32,
    segments: u32,
) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = vec![];

    let c = Vec3::from_slice(center);
    let du = 1.0 / segments as f32;
    let dv = 1.0 / rings as f32;

    for r in 0..=rings {
        let top_or_bot = r == 0 || r == rings;
        let n = if top_or_bot { segments - 1 } else { segments };
        for s in 0..=n {
            vertices.push(uv_sphere_vertex(&c, radius, r, s, du, dv, top_or_bot));
        }
    }

    let mut indices = vec![];

    let mut o1 = 0;
    let mut o2 = segments; // Top row has 1 less vertex because of single triangles.

    for r in 0..rings {
        debug!("r={r}, o1: {o1}, o2: {o2}");

        for s in 0..segments {
            if r == 0 {
                // Top triangles.
                indices.push(o1 + s);
                indices.push(o2 + s);
                indices.push(o2 + s + 1);
            } else if r > 0 && r < rings - 1 {
                // Ring quads (2 triangles).
                indices.push(o1 + s);
                indices.push(o2 + s);
                indices.push(o2 + s + 1);

                indices.push(o1 + s + 1);
                indices.push(o1 + s);
                indices.push(o2 + s + 1);
            } else {
                // Bottom triangles (r == rings - 1).
                indices.push(o1 + s + 1);
                indices.push(o1 + s);
                indices.push(o2 + s);
            }
        }

        o1 += if r == 0 { segments } else { segments + 1 }; // Top row as 1 less vertex.
        o2 = o1 + segments + 1; // We won't reach bottom row of vertices.
    }

    debug!(
        "Vertex count: {}, Indices count: {}",
        vertices.len(),
        indices.len()
    );

    /*
    debug!("-------------------------------------------------------------------------------");
    debug!("     Position                     Normal                       UV");
    debug!("-------------------------------------------------------------------------------");
    for (i, v) in vertices.iter().enumerate() {
        debug!(
            "{i: >3}  [{: >7.4}, {: >7.4}, {: >7.4}]  [{: >7.4}, {: >7.4}, {: >7.4}]  [{:.4}, {:.4}]",
            v.p[0], v.p[1], v.p[2], v.n[0], v.n[1], v.n[2], v.uv[0], v.uv[1],
        );
    }
    debug!("-------------------------------------------------------------------------------");
    debug!("Indices {indices:?}");
    debug!("-------------------------------------------------------------------------------");
    */

    (vertices, indices)
}

/// This will create a storage buffer to hold the mesh related data.
pub fn create_mesh_storage_buffer(
    vk: Arc<Vk>,
    meshes: &[Mesh],
    materials: &Materials,
) -> Result<Subbuffer<[closest_hit::Mesh]>> {
    let vertex_buffer_sizes = meshes.iter().map(|mesh| mesh.vertices.len());

    let index_buffer_sizes = meshes.iter().map(|mesh| mesh.indices.len());

    let materials = meshes.iter().map(|mesh| {
        // Material names are unique across all materials.
        if let Some(index) = materials.lambertian_material_indices.get(&mesh.material) {
            (MAT_TYPE_LAMBERTIAN, *index)
        } else if let Some(index) = materials.metal_material_indices.get(&mesh.material) {
            (MAT_TYPE_METAL, *index)
        } else if let Some(index) = materials.dielectric_material_indices.get(&mesh.material) {
            (MAT_TYPE_DIELECTRIC, *index)
        } else {
            warn!(
                "Mesh '{}' material '{}' not found",
                mesh.name, mesh.material
            );
            (MAT_TYPE_NONE, 0)
        }
    });

    let mesh_data: Vec<_> = vertex_buffer_sizes
        .zip(index_buffer_sizes)
        .zip(materials)
        .map(
            |((vertex_buffer_size, index_buffer_size), (material_type, material_index))| {
                closest_hit::Mesh {
                    vertexBufferSize: vertex_buffer_size as _,
                    indexBufferSize: index_buffer_size as _,
                    materialType: material_type,
                    materialIndex: material_index,
                }
            },
        )
        .collect();

    debug!("Creating mesh storage buffer");
    let buffer = Buffer::from_iter(
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
        mesh_data,
    )?;
    Ok(buffer)
}

/// Create a storage buffer for accessing vertices in shader code. This will pack vertices in order
/// of meshes.
pub fn create_mesh_vertex_buffer(
    vk: Arc<Vk>,
    meshes: &[Mesh],
) -> Result<Subbuffer<[closest_hit::MeshVertex]>> {
    let vertex_buffer_data: Vec<_> = meshes
        .iter()
        .flat_map(|mesh| mesh.vertices.iter().map(closest_hit::MeshVertex::from))
        .collect();

    debug!("Creating vertex buffer");
    let buffer = Buffer::from_iter(
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
        vertex_buffer_data,
    )?;
    Ok(buffer)
}

/// Create a storage buffer for accessing indices in shader code. This will pack indices in order
/// of meshes.
pub fn create_mesh_index_buffer(vk: Arc<Vk>, meshes: &[Mesh]) -> Result<Subbuffer<[u32]>> {
    let index_buffer_data: Vec<_> = meshes
        .iter()
        .flat_map(|mesh| mesh.indices.clone())
        .collect();

    debug!("Creating vertex buffer");
    let buffer = Buffer::from_iter(
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
        index_buffer_data,
    )?;
    Ok(buffer)
}
