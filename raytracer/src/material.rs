use std::{collections::HashMap, fmt, sync::Arc};

use anyhow::Result;
use log::debug;
use vulkano::buffer::{BufferUsage, Subbuffer};

use crate::{
    MaterialType, Vk, create_device_local_buffer, shaders::closest_hit, textures::Textures,
};

pub const MAT_TYPE_NONE: u32 = 0;
pub const MAT_TYPE_LAMBERTIAN: u32 = 1;
pub const MAT_TYPE_METAL: u32 = 2;
pub const MAT_TYPE_DIELECTRIC: u32 = 3;

pub const MAT_PROP_VALUE_TYPE_RGB: u32 = 0;
pub const MAT_PROP_VALUE_TYPE_IMAGE: u32 = 1;
pub const MAT_PROP_VALUE_TYPE_CHECKER: u32 = 2;

impl fmt::Debug for closest_hit::MaterialPropertyValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("closest_hit::MaterialPropertyValue")
            .field("propValueType", &self.propValueType)
            .field("index", &self.index)
            .finish()
    }
}

impl fmt::Debug for closest_hit::LambertianMaterial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("closest_hit::LambertianMaterial")
            .field("albedo", &self.albedo)
            .finish()
    }
}

impl fmt::Debug for closest_hit::MetalMaterial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("closest_hit::MetalMaterial")
            .field("albedo", &self.albedo)
            .field("fuzz", &self.fuzz)
            .finish()
    }
}

impl fmt::Debug for closest_hit::DielectricMaterial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("closest_hit::DielectricMaterial")
            .field("refractionIndex", &self.refractionIndex)
            .finish()
    }
}

#[derive(Debug)]
pub struct Materials {
    /// The lambertian materials. This will be used to create the storage buffers for shaders.
    pub lambertian_materials: Vec<closest_hit::LambertianMaterial>,

    /// The lambertian materials. This will be used to create the storage buffers for shaders.
    pub metal_materials: Vec<closest_hit::MetalMaterial>,

    /// The dielectric materials. This will be used to create the storage buffers for shaders.
    pub dielectric_materials: Vec<closest_hit::DielectricMaterial>,

    /// Maps unique lambertian materials to their index in `lambertian_materials`. These indices
    /// are used in the Mesh structure to be referenced in the storage buffers.
    pub lambertian_material_indices: HashMap<String, u32>,

    /// Maps unique metal materials to their index in `metal_materials`. These indices
    /// are used in the Mesh structure to be referenced in the storage buffers.
    pub metal_material_indices: HashMap<String, u32>,

    /// Maps unique dielectric materials to their index in `dielectric_materials`. These indices
    /// are used in the Mesh structure to be referenced in the storage buffers.
    pub dielectric_material_indices: HashMap<String, u32>,
}

impl Materials {
    pub fn new(materials: &[MaterialType], textures: &Textures) -> Self {
        let mut lambertian_materials = vec![];
        let mut metal_materials = vec![];
        let mut dielectric_materials = vec![];

        let mut lambertian_material_indices = HashMap::new();
        let mut metal_material_indices = HashMap::new();
        let mut dielectric_material_indices = HashMap::new();

        for material in materials.iter() {
            match material {
                MaterialType::Lambertian { name, albedo } => {
                    lambertian_material_indices
                        .insert(name.clone(), lambertian_materials.len() as _);

                    lambertian_materials.push(closest_hit::LambertianMaterial {
                        albedo: textures.to_shader(albedo).unwrap(),
                    });
                }
                MaterialType::Metal { name, albedo, fuzz } => {
                    metal_material_indices.insert(name.clone(), metal_materials.len() as _);

                    metal_materials.push(closest_hit::MetalMaterial {
                        albedo: textures.to_shader(albedo).unwrap(),
                        fuzz: textures.to_shader(fuzz).unwrap(),
                    });
                }
                MaterialType::Dielectric {
                    name,
                    refraction_index,
                } => {
                    dielectric_material_indices
                        .insert(name.clone(), dielectric_materials.len() as _);

                    dielectric_materials.push(closest_hit::DielectricMaterial {
                        refractionIndex: *refraction_index,
                    });
                }
            }
        }

        Materials {
            lambertian_materials,
            metal_materials,
            dielectric_materials,
            lambertian_material_indices,
            metal_material_indices,
            dielectric_material_indices,
        }
    }

    /// Create a storage buffers for accessing materials in shader code.
    pub fn create_buffers(&self, vk: Arc<Vk>) -> Result<MaterialBuffers> {
        let buffer_usage = BufferUsage::STORAGE_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS;

        // Note: We can't create buffers from empty list. So use a default material and push
        // constants will set the number of materials to 0 which the shader code checks for out of
        // bounds.

        debug!("Creating Lambertian materials buffer");
        let lambertian_materials_buffer = create_device_local_buffer(
            vk.clone(),
            buffer_usage,
            if !self.lambertian_materials.is_empty() {
                self.lambertian_materials.clone()
            } else {
                vec![closest_hit::LambertianMaterial {
                    albedo: closest_hit::MaterialPropertyValue {
                        propValueType: 0,
                        index: 0,
                    },
                }]
            },
        )?;

        debug!("Creating metal materials buffer");
        let metal_materials_buffer = create_device_local_buffer(
            vk.clone(),
            buffer_usage,
            if !self.metal_materials.is_empty() {
                self.metal_materials.clone()
            } else {
                vec![closest_hit::MetalMaterial {
                    albedo: closest_hit::MaterialPropertyValue {
                        propValueType: 0,
                        index: 0,
                    },
                    fuzz: closest_hit::MaterialPropertyValue {
                        propValueType: 0,
                        index: 0,
                    },
                }]
            },
        )?;

        debug!("Creating dielectric materials buffer");
        let dielectric_materials_buffer = create_device_local_buffer(
            vk.clone(),
            buffer_usage,
            if !self.dielectric_materials.is_empty() {
                self.dielectric_materials.clone()
            } else {
                vec![closest_hit::DielectricMaterial {
                    refractionIndex: 1.0,
                }]
            },
        )?;

        Ok(MaterialBuffers {
            lambertian: lambertian_materials_buffer,
            metal: metal_materials_buffer,
            dielectric: dielectric_materials_buffer,
        })
    }
}

/// Holds the storage buffers for the different material types.
pub struct MaterialBuffers {
    pub lambertian: Subbuffer<[closest_hit::LambertianMaterial]>,
    pub metal: Subbuffer<[closest_hit::MetalMaterial]>,
    pub dielectric: Subbuffer<[closest_hit::DielectricMaterial]>,
}
