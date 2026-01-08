use std::sync::Arc;

use anyhow::Result;
use log::{debug, info};
use scene_file::Primitive;
use shaders::any_hit;
use vulkano::{
    acceleration_structure::AabbPositions,
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
};

use crate::{MAT_TYPE_NONE, Materials, Vk, create_device_local_buffer};
#[derive(Copy, Clone, Debug)]
#[repr(u32)]
pub enum VolumeShape {
    Box = 0,
}

#[derive(Copy, Clone, Debug)]
#[repr(u32)]
pub enum MediumType {
    ConstantMedium = 0,
}

#[derive(Debug)]
pub struct Volume {
    pub name: String,
    pub aabb: [[f32; 3]; 2],
    pub shape: VolumeShape,
    pub material: String,
    pub medium_type: MediumType,
    pub medium_index: usize,
}

impl Volume {
    /// Create an AABB buffer for building the acceleration structure.
    pub fn create_blas_aabb_buffer(&self, vk: Arc<Vk>) -> Result<Subbuffer<[AabbPositions]>> {
        debug!("Creating BLAS AABB buffer");
        create_device_local_buffer(
            vk.clone(),
            BufferUsage::STORAGE_BUFFER
                | BufferUsage::SHADER_DEVICE_ADDRESS
                | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
            vec![AabbPositions {
                min: self.aabb[0],
                max: self.aabb[1],
            }],
        )
    }

    /// Return a mesh for primitives. If the primitive is not a mesh then it returns None.
    pub fn from_primitive(
        primitive: &Primitive,
        medium_type: MediumType,
        medium_index: usize,
    ) -> Option<Self> {
        let aabb = primitive.get_aabb();

        match primitive {
            Primitive::UvSphere { .. } => None,
            Primitive::Triangle { .. } => None,
            Primitive::Quad { .. } => None,
            Primitive::Box { name, material, .. } => Some(Self {
                name: name.clone(),
                aabb,
                material: material.clone(),
                shape: VolumeShape::Box,
                medium_type,
                medium_index,
            }),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ConstantMedium {
    pub density: f32,
}

impl ConstantMedium {
    pub fn new(density: f32) -> Self {
        Self { density }
    }
}

impl From<&ConstantMedium> for any_hit::ConstantMedium {
    fn from(value: &ConstantMedium) -> Self {
        Self {
            density: value.density,
            pad: [0.0, 0.0, 0.0],
        }
    }
}

/// This will create a storage buffer to hold the volume related data.
pub fn create_volume_storage_buffer(
    vk: Arc<Vk>,
    volumes: &[Arc<Volume>],
    materials: &Materials,
) -> Result<Subbuffer<[any_hit::Volume]>> {
    let volume_data: Vec<_> = volumes
        .iter()
        .map(|volume| {
            let type_and_index = materials.to_shader(&volume.material);
            if type_and_index.material_type == MAT_TYPE_NONE {
                info!(
                    "Volume '{}' material '{}' not found",
                    volume.name, volume.material
                );
            }
            any_hit::Volume {
                aabbMin: volume.aabb[0],
                aabbMax: volume.aabb[1],
                materialType: type_and_index.material_type,
                materialIndex: type_and_index.material_index,
                shape: volume.shape as _,
                mediumType: volume.medium_type as _,
                mediumIndex: volume.medium_index as _,
                pad: 0,
            }
        })
        .collect();

    debug!("Creating volume storage buffer");
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
        if !volume_data.is_empty() {
            volume_data
        } else {
            vec![any_hit::Volume {
                aabbMin: [0.0, 0.0, 0.0],
                aabbMax: [0.0, 0.0, 0.0],
                materialType: 0,
                materialIndex: 0,
                shape: 0,
                mediumType: 0,
                mediumIndex: 0,
                pad: 0,
            }]
        },
    )?;
    Ok(buffer)
}

/// This will create a storage buffer to hold the constant media related data.
pub fn create_constant_media_storage_buffer(
    vk: Arc<Vk>,
    constant_media: &[ConstantMedium],
) -> Result<Subbuffer<[any_hit::ConstantMedium]>> {
    let constant_medium_data: Vec<_> = constant_media.iter().map(|m| m.into()).collect();

    debug!("Creating constant media storage buffer");
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
        if !constant_medium_data.is_empty() {
            constant_medium_data
        } else {
            vec![any_hit::ConstantMedium {
                density: 1.0, // Avoid divide by zero.
                pad: [0.0, 0.0, 0.0],
            }]
        },
    )?;
    Ok(buffer)
}
