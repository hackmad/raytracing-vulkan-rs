use std::{f32::consts::PI, sync::Arc};

use anyhow::{Result, anyhow};
use glam::Vec3;
use log::{debug, info};
use scene_file::Primitive;
use shaders::ray_gen;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
};

use crate::{MAT_TYPE_NONE, Materials, MeshInstance, Transform, Vk, create_device_local_buffer};

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

impl From<&Vertex> for ray_gen::MeshVertex {
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
    ) -> Result<Subbuffer<[ray_gen::MeshVertex]>> {
        debug!("Creating BLAS vertex buffer");
        create_device_local_buffer(
            vk.clone(),
            BufferUsage::VERTEX_BUFFER
                | BufferUsage::SHADER_DEVICE_ADDRESS
                | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
            self.vertices.iter().map(ray_gen::MeshVertex::from),
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
    meshes: &[Arc<Mesh>],
    materials: &Materials,
) -> Result<Subbuffer<[ray_gen::Mesh]>> {
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
                ray_gen::Mesh {
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
    meshes: &[Arc<Mesh>],
) -> Result<Subbuffer<[ray_gen::MeshVertex]>> {
    let vertex_buffer_data: Vec<_> = meshes
        .iter()
        .flat_map(|mesh| mesh.vertices.iter().map(ray_gen::MeshVertex::from))
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
pub fn create_mesh_index_buffer(vk: Arc<Vk>, meshes: &[Arc<Mesh>]) -> Result<Subbuffer<[u32]>> {
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

struct Area {
    value: f32,
    mesh_index: usize,
    primitive_index: usize,
}

pub struct LightSourceAliasTable {
    pub buffer: Subbuffer<[ray_gen::LightSourceAliasTableEntry]>,
    pub triangle_count: usize,
    pub total_area: f32,
}

/// Builds a CDF of triangle areas computed in world space for Vose's alias method.
/// See https://en.wikipedia.org/wiki/Alias_method.
///
/// The areas will be used to sample triangles that are part of meshes used as light sources.
pub fn create_light_source_alias_table(
    vk: Arc<Vk>,
    mesh_instances: &[MeshInstance],
    meshes: &[Arc<Mesh>],
    materials: &Materials,
) -> Result<LightSourceAliasTable> {
    let light_sources: Vec<_> = mesh_instances
        .iter()
        .filter(|mesh_instance| {
            materials
                .diffuse_light_material_indices
                .contains_key(&meshes[mesh_instance.mesh_index].material)
        })
        .collect();

    let light_count = light_sources.len();

    let mut world_space_areas = Vec::with_capacity(1024);

    for light_source in light_sources {
        let mesh = meshes[light_source.mesh_index].as_ref();
        let indices = mesh.indices.as_slice();
        let vertices = mesh.vertices.as_slice();

        for i in (0..mesh.indices.len()).step_by(3) {
            let primitive_index = i / 3;

            let indices = [
                indices[i] as usize,
                indices[i + 1] as usize,
                indices[i + 2] as usize,
            ];

            let light_object_to_world = match light_source.object_to_world {
                Transform::Static(ref t) => Ok(t.to_mat4()),
                Transform::Animated { .. } => Err(anyhow!(
                    "Animated transform for light sources not implemented"
                )),
            }?;

            let p = indices.map(|i| {
                let v = vertices[i].p;
                let v4 = [v[0], v[1], v[2], 1.0].into();
                let w = light_object_to_world.mul_vec4(v4);
                Vec3::new(w.x, w.y, w.z)
            });

            let v0 = p[1] - p[0];
            let v1 = p[2] - p[0];
            let area = 0.5 * v0.cross(v1).length();

            // Discard degenerate triangles
            if area > 1e-8 {
                world_space_areas.push(Area {
                    value: area,
                    mesh_index: light_source.mesh_index,
                    primitive_index,
                });
            }
        }
    }

    let triangle_count = world_space_areas.len();

    let (alias_table, total_area) = if triangle_count > 0 {
        let (table, total) = build_alias_table(&world_space_areas);
        debug_assert!(table.len() == triangle_count, "Alias table size mismatch");
        (table, total)
    } else {
        // Use dummy table so descriptor set can be built without crashing.
        // The count will be 0 which should be used to check GPU-side to
        // not do light sampling if we do not have a table to use.
        let table = vec![ray_gen::LightSourceAliasTableEntry {
            probability: 0.0,
            alias: 0,
            meshId: 0,
            primitiveId: 0,
        }];
        (table, 0.0)
    };

    debug!(
        "Creating buffer for light source alias table: {} lights, total area: {}, {} triangles with non-zero area",
        light_count, total_area, triangle_count
    );
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
        alias_table,
    )?;

    Ok(LightSourceAliasTable {
        buffer,
        triangle_count,
        total_area,
    })
}

fn build_alias_table(areas: &[Area]) -> (Vec<ray_gen::LightSourceAliasTableEntry>, f32) {
    let n = areas.len();
    let total_area = areas
        .iter()
        .fold(0.0_f64, |acc, area| acc + area.value as f64) as f32;

    let mut q = vec![0.0; n];
    for i in 0..n {
        q[i] = areas[i].value * n as f32 / total_area;
    }

    let mut small = Vec::new();
    let mut large = Vec::new();

    for (i, v) in q.iter().enumerate() {
        if *v < 1.0 {
            small.push(i);
        } else {
            large.push(i);
        }
    }

    let mut probabilities = vec![0.0; n];
    let mut aliases = vec![0u32; n];

    while let (Some(s), Some(l)) = (small.pop(), large.pop()) {
        probabilities[s] = q[s];
        aliases[s] = l as u32;

        q[l] -= 1.0 - q[s];

        if q[l] < 1.0 {
            small.push(l);
        } else {
            large.push(l);
        }
    }

    for i in small.into_iter().chain(large.into_iter()) {
        probabilities[i] = 1.0;
        aliases[i] = i as u32;
    }

    let alias_table = probabilities
        .iter()
        .zip(aliases.iter())
        .enumerate()
        .map(
            |(i, (probability, alias))| ray_gen::LightSourceAliasTableEntry {
                probability: *probability,
                alias: *alias,
                meshId: areas[i].mesh_index as _,
                primitiveId: areas[i].primitive_index as _,
            },
        )
        .collect();

    (alias_table, total_area)
}
