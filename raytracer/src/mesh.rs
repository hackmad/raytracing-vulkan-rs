use std::{f32::consts::PI, sync::Arc};

use anyhow::Result;
use glam::Vec3;
use log::{debug, info};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
};

use crate::{
    MAT_TYPE_NONE, Materials, Primitive, Vk, create_device_local_buffer, shaders::closest_hit,
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

impl From<&Primitive> for Mesh {
    fn from(value: &Primitive) -> Self {
        match value {
            Primitive::UvSphere {
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

            Primitive::Triangle {
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

            Primitive::Quad {
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

            Primitive::Box {
                name,
                corners,
                material,
            } => {
                let (vertices, indices) = generate_box(corners);
                Mesh {
                    name: name.clone(),
                    vertices,
                    indices,
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

fn uv_rect(col: usize, row: usize, cols: usize, rows: usize) -> [[f32; 2]; 4] {
    let cell_w = 1.0 / cols as f32;
    let cell_h = 1.0 / rows as f32;

    let u0 = col as f32 * cell_w;
    let v0 = 1.0 - (row as f32 + 1.0) * cell_h; // flip V: 0 at top
    let u1 = u0 + cell_w;
    let v1 = v0 + cell_h;

    [
        [u0, v1], // BL
        [u1, v1], // BR
        [u1, v0], // TR
        [u0, v0], // TL
    ]
}

#[rustfmt::skip]
fn generate_box(corners: &[[f32; 3]; 2]) -> (Vec<Vertex>, Vec<u32>) {
    let a = Vec3::from_slice(&corners[0]);
    let b = Vec3::from_slice(&corners[1]);

    let [x0, y0, z0] = a.min(b).to_array();
    let [x1, y1, z1] = a.max(b).to_array();

    let (lx, hx) = (x0, x1);
    let (ly, hy) = (y0, y1);
    let (lz, hz) = (z0, z1);

    let uv_front =  uv_rect(1, 1, 4, 3);
    let uv_back =   uv_rect(3, 1, 4, 3);
    let uv_left =   uv_rect(0, 1, 4, 3);
    let uv_right =  uv_rect(2, 1, 4, 3);
    let uv_top =    uv_rect(1, 0, 4, 3);
    let uv_bottom = uv_rect(1, 2, 4, 3);

    let vertices = vec![
        // Front (+Z)
        Vertex::new([lx, ly, hz], [ 0.0,  0.0,  1.0],  uv_front[0]),
        Vertex::new([hx, ly, hz], [ 0.0,  0.0,  1.0],  uv_front[1]),
        Vertex::new([hx, hy, hz], [ 0.0,  0.0,  1.0],  uv_front[2]),
        Vertex::new([lx, hy, hz], [ 0.0,  0.0,  1.0],  uv_front[3]),

        // Back (-Z)
        Vertex::new([hx, ly, lz], [ 0.0,  0.0, -1.0],   uv_back[0]),
        Vertex::new([lx, ly, lz], [ 0.0,  0.0, -1.0],   uv_back[1]),
        Vertex::new([lx, hy, lz], [ 0.0,  0.0, -1.0],   uv_back[2]),
        Vertex::new([hx, hy, lz], [ 0.0,  0.0, -1.0],   uv_back[3]),
                                                                 
        // Left (-X)                                             
        Vertex::new([lx, ly, lz], [-1.0,  0.0,  0.0],   uv_left[0]),
        Vertex::new([lx, ly, hz], [-1.0,  0.0,  0.0],   uv_left[1]),
        Vertex::new([lx, hy, hz], [-1.0,  0.0,  0.0],   uv_left[2]),
        Vertex::new([lx, hy, lz], [-1.0,  0.0,  0.0],   uv_left[3]),
                                                                 
        // Right (+X)                                            
        Vertex::new([hx, ly, hz], [ 1.0,  0.0,  0.0],  uv_right[0]),
        Vertex::new([hx, ly, lz], [ 1.0,  0.0,  0.0],  uv_right[1]),
        Vertex::new([hx, hy, lz], [ 1.0,  0.0,  0.0],  uv_right[2]),
        Vertex::new([hx, hy, hz], [ 1.0,  0.0,  0.0],  uv_right[3]),

        // Top (-Y)
        Vertex::new([lx, hy, hz], [ 0.0, -1.0,  0.0],    uv_top[0]),
        Vertex::new([hx, hy, hz], [ 0.0, -1.0,  0.0],    uv_top[1]),
        Vertex::new([hx, hy, lz], [ 0.0, -1.0,  0.0],    uv_top[2]),
        Vertex::new([lx, hy, lz], [ 0.0, -1.0,  0.0],    uv_top[3]),

        // Bottom (+Y)
        Vertex::new([lx, ly, lz], [ 0.0,  1.0,  0.0], uv_bottom[0]),
        Vertex::new([hx, ly, lz], [ 0.0,  1.0,  0.0], uv_bottom[1]),
        Vertex::new([hx, ly, hz], [ 0.0,  1.0,  0.0], uv_bottom[2]),
        Vertex::new([lx, ly, hz], [ 0.0,  1.0,  0.0], uv_bottom[3]),
    ];

    // 6 faces, each with 2 triangles = 6 indices per face
    let indices = vec![
        // Front
        0, 1, 2,
        2, 3, 0,

        // Back
        4, 5, 6,
        6, 7, 4,

        // Left
        8, 9, 10,
        10, 11, 8,

        // Right
        12, 13, 14,
        14, 15, 12,

        // Top
        16, 17, 18,
        18, 19, 16,

        // Bottom
        20, 21, 22,
        22, 23, 20,
    ];

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
        let type_and_index = materials.to_shader(&mesh.material);
        if type_and_index.material_type == MAT_TYPE_NONE {
            info!(
                "Mesh '{}' material '{}' not found",
                mesh.name, mesh.material
            );
        }
        (type_and_index.material_type, type_and_index.material_index)
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
