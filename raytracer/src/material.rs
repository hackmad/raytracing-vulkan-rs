use std::{collections::HashMap, sync::Arc};

use anyhow::Result;
use ash::vk;
use log::debug;
use scene_file::Material;
use shaders::{
    self, MAT_TYPE_DIELECTRIC, MAT_TYPE_DIFFUSE_LIGHT, MAT_TYPE_LAMBERTIAN, MAT_TYPE_METAL,
    MAT_TYPE_NONE,
};
use vulkan::{Buffer, VulkanContext};

use crate::textures::Textures;

#[derive(Debug)]
pub struct Materials {
    /// The lambertian materials. This will be used to create the storage buffers for shaders.
    pub lambertian_materials: Vec<shaders::LambertianMaterial>,

    /// The lambertian materials. This will be used to create the storage buffers for shaders.
    pub metal_materials: Vec<shaders::MetalMaterial>,

    /// The dielectric materials. This will be used to create the storage buffers for shaders.
    pub dielectric_materials: Vec<shaders::DielectricMaterial>,

    /// The diffuse light materials. This will be used to create the storage buffers for shaders.
    pub diffuse_light_materials: Vec<shaders::DiffuseLightMaterial>,

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

                    lambertian_materials.push(shaders::LambertianMaterial {
                        albedo: textures.to_shader(albedo).unwrap(),
                    });
                }
                Material::Metal { name, albedo, fuzz } => {
                    metal_material_indices.insert(name.clone(), metal_materials.len() as _);

                    metal_materials.push(shaders::MetalMaterial {
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

                    dielectric_materials.push(shaders::DielectricMaterial {
                        refraction_index: *refraction_index,
                    });
                }
                Material::DiffuseLight { name, emit } => {
                    diffuse_light_material_indices
                        .insert(name.clone(), diffuse_light_materials.len() as _);

                    diffuse_light_materials.push(shaders::DiffuseLightMaterial {
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
    pub fn create_buffers(&self, context: Arc<VulkanContext>) -> Result<MaterialBuffers> {
        let buffer_usage =
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;

        // Note: We can't create buffers from empty list. So use a default material and push
        // constants will set the number of materials to 0 which the shader code checks for out of
        // bounds.

        debug!("Creating Lambertian materials buffer");
        let lambertian_materials_buffer = Buffer::new_device_local_storage_buffer(
            context.clone(),
            buffer_usage,
            &self.lambertian_materials,
        )?;

        debug!("Creating metal materials buffer");
        let metal_materials_buffer = Buffer::new_device_local_storage_buffer(
            context.clone(),
            buffer_usage,
            &self.metal_materials,
        )?;

        debug!("Creating dielectric materials buffer");
        let dielectric_materials_buffer = Buffer::new_device_local_storage_buffer(
            context.clone(),
            buffer_usage,
            &self.dielectric_materials,
        )?;

        debug!("Creating diffuse light materials buffer");
        let diffuse_light_materials_buffer = Buffer::new_device_local_storage_buffer(
            context.clone(),
            buffer_usage,
            &self.diffuse_light_materials,
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
    pub lambertian: Buffer,
    pub metal: Buffer,
    pub dielectric: Buffer,
    pub diffuse_light: Buffer,
}
