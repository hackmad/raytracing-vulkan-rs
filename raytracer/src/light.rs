use std::sync::Arc;

use anyhow::{Result, anyhow};
use glam::Vec3;
use log::debug;
use shaders::ray_gen;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
};

use crate::{Materials, Mesh, MeshInstance, Transform, Vk};

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
