use std::{f32::consts::PI, fmt, sync::Arc};

use anyhow::{Result, anyhow};
use log::warn;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
};

use crate::raytracer::{
    MAT_TYPE_DIELECTRIC, MAT_TYPE_LAMBERTIAN, MAT_TYPE_METAL, MAT_TYPE_NONE, Materials, ObjectType,
    Vk, create_device_local_buffer, shaders::closest_hit,
};

impl fmt::Debug for closest_hit::MeshVertex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("closest_hit::MeshVertex")
            .field("position", &self.position)
            .field("normal", &self.normal)
            .field("texCoord", &self.texCoord)
            .finish()
    }
}

#[derive(Debug)]
pub struct Mesh {
    pub name: String,
    pub vertices: Vec<closest_hit::MeshVertex>,
    pub indices: Vec<u32>,
    pub material: String,
}

impl Mesh {
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
        }
    }
}

fn generate_uv_sphere(
    center: &[f32; 3],
    radius: f32,
    rings: u32,
    segments: u32,
) -> (Vec<closest_hit::MeshVertex>, Vec<u32>) {
    let mut vertices = vec![];
    let mut indices = vec![];

    let c = glam::Vec3::from_slice(center);

    // Top vertex
    let p = glam::Vec3::new(0.0, radius, 0.0);
    let n = glam::Vec3::new(0.0, 1.0, 0.0);
    vertices.push(closest_hit::MeshVertex {
        position: (p + c).to_array(),
        normal: n.to_array(),
        texCoord: [0.0, 0.0],
    });
    let i_top = 0_u32;

    // Vertices per ring / segment.
    for i in 0..(rings - 1) {
        let phi = PI * (i + 1) as f32 / rings as f32;
        for j in 0..segments {
            let theta = 2.0 * PI * j as f32 / segments as f32;
            let x = radius * phi.sin() * theta.cos();
            let y = radius * phi.cos();
            let z = radius * phi.sin() * theta.sin();

            let p = glam::Vec3::new(x, y, z);
            let n = p.normalize();

            let u = n.x.atan2(n.z) / (2.0 * PI) + 0.5;
            let v = n.y * 0.5 + 0.5;

            vertices.push(closest_hit::MeshVertex {
                position: (p + c).to_array(),
                normal: n.to_array(),
                texCoord: [u, v],
            });
        }
    }

    // Bottom vertex
    let p = glam::Vec3::new(0.0, -radius, 0.0);
    let n = glam::Vec3::new(0.0, -1.0, 0.0);
    vertices.push(closest_hit::MeshVertex {
        position: (p + c).to_array(),
        normal: n.to_array(),
        texCoord: [0.0, 1.0],
    });
    let i_bottom = (vertices.len() - 1) as u32;

    // Top and bottom triangles.
    for i in 0..segments {
        indices.push(i_top);
        indices.push(i + 1);
        indices.push((i + 1) % segments + 1);

        indices.push(i_bottom);
        indices.push(i + segments * (rings - 2) + 1);
        indices.push((i + 1) % segments + segments * (rings - 2) + 1);
    }

    // Ring triangles.
    for j in 0..(rings - 2) {
        let j0 = j * segments + 1;
        let j1 = (j + 1) * segments + 1;
        for i in 0..segments {
            let i0 = j0 + i;
            let i1 = j0 + (i + 1) % segments;
            let i2 = j1 + (i + 1) % segments;
            let i3 = j1 + i;
            indices.push(i0);
            indices.push(i1);
            indices.push(i2);

            indices.push(i0);
            indices.push(i2);
            indices.push(i3);
        }
    }

    (vertices, indices)
}

/// This will create 2 storage buffers that can be accessed by their device address only by the GPU for the vertices
/// and indices. These addresses will be packed in another storage buffer representing the mesh data which will be
/// returned.
pub fn create_mesh_storage_buffer(
    vk: Arc<Vk>,
    meshes: &[Mesh],
    materials: &Materials,
) -> Result<Subbuffer<[closest_hit::Mesh]>> {
    let vertices_storage_buffers = meshes
        .iter()
        .map(|mesh| mesh.create_vertices_storage_buffer(vk.clone()))
        .collect::<Result<Vec<_>>>()?;

    let indices_storage_buffers = meshes
        .iter()
        .map(|mesh| mesh.create_indices_storage_buffer(vk.clone()))
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

    let material_types_and_indices: Vec<_> = meshes
        .iter()
        .map(|mesh| {
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
        })
        .collect();

    let mesh_data = vertices_buffer_device_addresses
        .into_iter()
        .zip(indices_buffer_device_addresses)
        .zip(material_types_and_indices)
        .map(
            |((vertices_ref, indices_ref), (material_type, material_index))| closest_hit::Mesh {
                verticesRef: vertices_ref,
                indicesRef: indices_ref,
                materialType: material_type,
                materialIndex: material_index,
            },
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
        mesh_data,
    )?;

    Ok(buffer)
}
