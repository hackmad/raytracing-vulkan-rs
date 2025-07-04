use std::{collections::HashMap, sync::Arc};

use anyhow::Result;
use log::debug;
use scene_file::Material;
use shaders::closest_hit;
use vulkano::buffer::{BufferUsage, Subbuffer};

use crate::{Vk, create_device_local_buffer, textures::Textures};

// NOTE: Update Materials::to_shader() when adding new materials.
pub const MAT_TYPE_NONE: u32 = 0;
pub const MAT_TYPE_LAMBERTIAN: u32 = 1;
pub const MAT_TYPE_METAL: u32 = 2;
pub const MAT_TYPE_DIELECTRIC: u32 = 3;
pub const MAT_TYPE_DIFFUSE_LIGHT: u32 = 4;

pub const MAT_PROP_VALUE_TYPE_RGB: u32 = 0;
pub const MAT_PROP_VALUE_TYPE_IMAGE: u32 = 1;
pub const MAT_PROP_VALUE_TYPE_CHECKER: u32 = 2;
pub const MAT_PROP_VALUE_TYPE_NOISE: u32 = 3;

#[derive(Debug)]
pub struct Materials {
    /// The lambertian materials. This will be used to create the storage buffers for shaders.
    pub lambertian_materials: Vec<closest_hit::LambertianMaterial>,

    /// The lambertian materials. This will be used to create the storage buffers for shaders.
    pub metal_materials: Vec<closest_hit::MetalMaterial>,

    /// The dielectric materials. This will be used to create the storage buffers for shaders.
    pub dielectric_materials: Vec<closest_hit::DielectricMaterial>,

    /// The diffuse light materials. This will be used to create the storage buffers for shaders.
    pub diffuse_light_materials: Vec<closest_hit::DiffuseLightMaterial>,

    /// Maps unique lambertian materials to their index in `lambertian_materials`. These indices
    /// are used in the Mesh structure to be referenced in the storage buffers.
    pub lambertian_material_indices: HashMap<String, u32>,

    /// Maps unique metal materials to their index in `metal_materials`. These indices
    /// are used in the Mesh structure to be referenced in the storage buffers.
    pub metal_material_indices: HashMap<String, u32>,

    /// Maps unique dielectric materials to their index in `dielectric_materials`. These indices
    /// are used in the Mesh structure to be referenced in the storage buffers.
    pub dielectric_material_indices: HashMap<String, u32>,

    /// Maps unique diffuse light materials to their index in `diffuse_light_materials`. These indices
    /// are used in the Mesh structure to be referenced in the storage buffers.
    pub diffuse_light_material_indices: HashMap<String, u32>,
}

impl Materials {
    pub fn new(materials: &[Material], textures: &Textures) -> Self {
        let mut lambertian_materials = vec![];
        let mut metal_materials = vec![];
        let mut dielectric_materials = vec![];
        let mut diffuse_light_materials = vec![];

        let mut lambertian_material_indices = HashMap::new();
        let mut metal_material_indices = HashMap::new();
        let mut dielectric_material_indices = HashMap::new();
        let mut diffuse_light_material_indices = HashMap::new();

        for material in materials.iter() {
            match material {
                Material::Lambertian { name, albedo } => {
                    lambertian_material_indices
                        .insert(name.clone(), lambertian_materials.len() as _);

                    lambertian_materials.push(closest_hit::LambertianMaterial {
                        albedo: textures.to_shader(albedo).unwrap(),
                    });
                }
                Material::Metal { name, albedo, fuzz } => {
                    metal_material_indices.insert(name.clone(), metal_materials.len() as _);

                    metal_materials.push(closest_hit::MetalMaterial {
                        albedo: textures.to_shader(albedo).unwrap(),
                        fuzz: textures.to_shader(fuzz).unwrap(),
                    });
                }
                Material::Dielectric {
                    name,
                    refraction_index,
                } => {
                    dielectric_material_indices
                        .insert(name.clone(), dielectric_materials.len() as _);

                    dielectric_materials.push(closest_hit::DielectricMaterial {
                        refractionIndex: *refraction_index,
                    });
                }
                Material::DiffuseLight { name, emit } => {
                    diffuse_light_material_indices
                        .insert(name.clone(), diffuse_light_materials.len() as _);

                    diffuse_light_materials.push(closest_hit::DiffuseLightMaterial {
                        emit: textures.to_shader(emit).unwrap(),
                    });
                }
            }
        }

        Materials {
            lambertian_materials,
            metal_materials,
            dielectric_materials,
            diffuse_light_materials,
            lambertian_material_indices,
            metal_material_indices,
            dielectric_material_indices,
            diffuse_light_material_indices,
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

        debug!("Creating diffuse light materials buffer");
        let diffuse_light_materials_buffer = create_device_local_buffer(
            vk.clone(),
            buffer_usage,
            if !self.diffuse_light_materials.is_empty() {
                self.diffuse_light_materials.clone()
            } else {
                vec![closest_hit::DiffuseLightMaterial {
                    emit: closest_hit::MaterialPropertyValue {
                        propValueType: 0,
                        index: 0,
                    },
                }]
            },
        )?;

        Ok(MaterialBuffers {
            lambertian: lambertian_materials_buffer,
            metal: metal_materials_buffer,
            dielectric: dielectric_materials_buffer,
            diffuse_light: diffuse_light_materials_buffer,
        })
    }

    pub fn to_shader(&self, material: &str) -> MaterialAndIndex {
        // Material names are unique across all materials.
        if let Some(index) = self.lambertian_material_indices.get(material) {
            MaterialAndIndex::new(MAT_TYPE_LAMBERTIAN, *index)
        } else if let Some(index) = self.metal_material_indices.get(material) {
            MaterialAndIndex::new(MAT_TYPE_METAL, *index)
        } else if let Some(index) = self.dielectric_material_indices.get(material) {
            MaterialAndIndex::new(MAT_TYPE_DIELECTRIC, *index)
        } else if let Some(index) = self.diffuse_light_material_indices.get(material) {
            MaterialAndIndex::new(MAT_TYPE_DIFFUSE_LIGHT, *index)
        } else {
            MaterialAndIndex::new(MAT_TYPE_NONE, 0)
        }
    }
}

pub struct MaterialAndIndex {
    pub material_type: u32,
    pub material_index: u32,
}

impl MaterialAndIndex {
    pub fn new(material_type: u32, material_index: u32) -> Self {
        Self {
            material_type,
            material_index,
        }
    }
}

/// Holds the storage buffers for the different material types.
pub struct MaterialBuffers {
    pub lambertian: Subbuffer<[closest_hit::LambertianMaterial]>,
    pub metal: Subbuffer<[closest_hit::MetalMaterial]>,
    pub dielectric: Subbuffer<[closest_hit::DielectricMaterial]>,
    pub diffuse_light: Subbuffer<[closest_hit::DiffuseLightMaterial]>,
}
