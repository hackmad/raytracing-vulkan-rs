use std::sync::Arc;

use anyhow::Result;
use log::{debug, info};
use scene_file::Primitive;
use shaders::{any_hit, intersection};
use vulkano::{
    acceleration_structure::AabbPositions,
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
};

use crate::{MAT_TYPE_NONE, Materials, Vk, create_device_local_buffer};

#[derive(Copy, Clone, Debug)]
pub enum VolumeShape {
    Box,
    Sphere { center: [f32; 3], radius: f32 },
}

impl VolumeShape {
    pub fn to_shader(&self) -> u32 {
        match self {
            Self::Box => 0,           // VOLUME_SHAPE_BOX
            Self::Sphere { .. } => 1, // VOLUME_SHAPE_SPHERE
        }
    }
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
            Primitive::UvSphere {
                name,
                center,
                radius,
                ..
            } => Some(Self {
                name: name.clone(),
                aabb,
                shape: VolumeShape::Sphere {
                    center: *center,
                    radius: *radius,
                },
                medium_type,
                medium_index,
            }),
            Primitive::Triangle { .. } => None,
            Primitive::Quad { .. } => None,
            Primitive::Box { name, .. } => Some(Self {
                name: name.clone(),
                aabb,
                shape: VolumeShape::Box,
                medium_type,
                medium_index,
            }),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ConstantMedium {
    pub density: f32,
    pub phase_function: String,
}

impl ConstantMedium {
    pub fn new(density: f32, phase_function: String) -> Self {
        Self {
            density,
            phase_function,
        }
    }
}

pub struct VolumeStorageBuffers {
    pub volume_buffer: Subbuffer<[intersection::Volume]>,
    pub sphere_volume_buffer: Subbuffer<[intersection::SphereVolume]>,
}

/// This will create a storage buffer to hold the volume related data.
pub fn create_volume_storage_buffers(
    vk: Arc<Vk>,
    volumes: &[Arc<Volume>],
) -> Result<VolumeStorageBuffers> {
    let mut volume_data: Vec<intersection::Volume> = Vec::new();
    let mut sphere_volume_data: Vec<intersection::SphereVolume> = Vec::new();

    for volume in volumes.iter() {
        // - For all volumes we will create intersection::Volume that has AABB info.
        // - For box volumes, there is no additional metadata needed because the AABB corners define the box and
        //   shapeVolumeIndex is ignored.
        // - For volume shapes other than box, we create corresponding intersection::***Volume. and set
        //   shapeVolumeIndex to point to it.

        let shape_volume_index = match volume.shape {
            VolumeShape::Box => 0,

            VolumeShape::Sphere { center, radius } => {
                let sphere_volume = intersection::SphereVolume { center, radius };
                sphere_volume_data.push(sphere_volume);
                sphere_volume_data.len() - 1
            }
        };

        let volume = intersection::Volume {
            aabbMin: volume.aabb[0],
            aabbMax: volume.aabb[1],
            shape: volume.shape.to_shader(),
            mediumType: volume.medium_type as _,
            mediumIndex: volume.medium_index as _,
            shapeVolumeIndex: shape_volume_index as _,
            pad: [0.0, 0.0],
        };
        volume_data.push(volume);
    }

    debug!("Creating volume storage buffer");
    let volume_buffer = Buffer::from_iter(
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
            vec![intersection::Volume {
                aabbMin: [0.0, 0.0, 0.0],
                aabbMax: [0.0, 0.0, 0.0],
                shape: 0,
                mediumType: 0,
                mediumIndex: 0,
                shapeVolumeIndex: 0,
                pad: [0.0, 0.0],
            }]
        },
    )?;

    debug!("Creating sphere volume storage buffer");
    let sphere_volume_buffer = Buffer::from_iter(
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
        if !sphere_volume_data.is_empty() {
            sphere_volume_data
        } else {
            vec![intersection::SphereVolume {
                center: [0.0, 0.0, 0.0],
                radius: 1.0,
            }]
        },
    )?;

    Ok(VolumeStorageBuffers {
        volume_buffer,
        sphere_volume_buffer,
    })
}

/// This will create a storage buffer to hold the constant media related data.
pub fn create_constant_media_storage_buffer(
    vk: Arc<Vk>,
    constant_media: &[ConstantMedium],
    materials: &Materials,
) -> Result<Subbuffer<[any_hit::ConstantMedium]>> {
    let constant_medium_data: Vec<_> = constant_media
        .iter()
        .map(|m| {
            let type_and_index = materials.to_shader(&m.phase_function);
            if type_and_index.material_type == MAT_TYPE_NONE {
                info!("Constant density material '{}' not found", m.phase_function);
            }

            any_hit::ConstantMedium {
                density: m.density,
                materialType: type_and_index.material_type,
                materialIndex: type_and_index.material_index,
                pad: 0,
            }
        })
        .collect();

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
                materialType: 0,
                materialIndex: 0,
                pad: 0,
            }]
        },
    )?;
    Ok(buffer)
}
