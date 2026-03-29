use std::{collections::HashMap, sync::Arc};

use anyhow::{Result, anyhow};
use scene_file::{InstanceType, SceneFile};

use crate::{Mesh, Transform};

#[derive(Debug)]
pub struct ConstantMedium {
    pub name: String,
    pub density: f32,
    pub phase_function: String,
}
impl ConstantMedium {
    fn new(name: &str, density: f32, phase_function: &str) -> Self {
        Self {
            name: name.to_string(),
            density,
            phase_function: phase_function.to_string(),
        }
    }
}

/// Stores instance related data.
#[derive(Debug)]
pub struct Instance {
    /// Index of the mesh or volume.
    /// Use gl_InstanceCustomIndexEXT in shader code to retrieve it.
    /// Volume indices will be packed after mesh indices.
    pub index: usize,

    /// Transformation for this instance.
    pub object_to_world: Transform,

    /// Type of instance.
    pub instance_type: InstanceType,
}

impl Instance {
    /// Create a new instance with a given index and object-to-world transformation.
    pub fn new(index: usize, instance_type: InstanceType, object_to_world: Transform) -> Self {
        Self {
            index,
            object_to_world,
            instance_type,
        }
    }

    /// Returns the 3x4 matrix used in Vulkan transformations for acceleration structures.
    /// For animated transforms, it interpolates the transformation for time in [0, 1].
    pub fn get_vulkan_acc_transform(&self, time: f32) -> [[f32; 4]; 3] {
        match self.object_to_world {
            Transform::Static(ref t) => t.to_vulkan_acc_mat(),
            Transform::Animated {
                start: ref t0,
                end: ref t1,
            } => t0.lerp(t1, time).to_vulkan_acc_mat(),
        }
    }
}

pub struct Instances {
    pub meshes: Vec<Arc<Mesh>>,
    pub mesh_name_to_index: HashMap<String, usize>,
    pub instances: Vec<Instance>,
    pub constant_media: Vec<ConstantMedium>,
    pub constant_medium_name_to_index: HashMap<String, usize>,
}

impl Instances {
    pub fn new(scene_file: &SceneFile) -> Result<Self> {
        let mut meshes: Vec<Arc<Mesh>> = Vec::new();
        let mut mesh_name_to_index: HashMap<String, usize> = HashMap::new();
        let mut constant_media: Vec<ConstantMedium> = Vec::new();
        let mut constant_medium_name_to_index: HashMap<String, usize> = HashMap::new();

        for instance in scene_file.instances.iter() {
            let primitive = match scene_file
                .primitives
                .iter()
                .find(|p| p.get_name() == instance.name)
            {
                Some(p) => p,
                None => {
                    return Err(anyhow!(
                        "Primitive not found for instance {}",
                        instance.name
                    ));
                }
            };

            // Both surfaces and volumes have meshes. In latter case the mesh is defining
            // the volume boundary.
            let mesh = Arc::new(Mesh::from(primitive));
            mesh_name_to_index.insert(instance.name.clone(), meshes.len());
            meshes.push(mesh);

            match &instance.instance_type {
                InstanceType::Surface => {
                    // Nothing else to do
                }

                InstanceType::ConstantMedium {
                    density,
                    phase_function,
                } => {
                    constant_medium_name_to_index
                        .insert(instance.name.clone(), constant_media.len());
                    constant_media.push(ConstantMedium::new(
                        &instance.name,
                        *density,
                        phase_function,
                    ));
                }
            }
        }

        // Get instances.
        let mut instances: Vec<Instance> = Vec::new();
        for instance in scene_file.instances.iter() {
            let object_to_world = instance.get_object_to_world_space_matrix();
            let transform = Transform::from(object_to_world);

            match &instance.instance_type {
                InstanceType::Surface => {
                    if let Some(index) = mesh_name_to_index.get(&instance.name) {
                        instances.push(Instance::new(*index, InstanceType::Surface, transform));
                    } else {
                        return Err(anyhow!("Mesh {} not found", instance.name));
                    }
                }

                InstanceType::ConstantMedium {
                    density,
                    phase_function,
                } => {
                    if let Some(index) = constant_medium_name_to_index.get(&instance.name) {
                        instances.push(Instance::new(
                            *index,
                            InstanceType::ConstantMedium {
                                density: *density,
                                phase_function: phase_function.clone(),
                            },
                            transform,
                        ));
                    } else {
                        return Err(anyhow!("Volume {} not found", instance.name));
                    }
                }
            }
        }

        Ok(Self {
            meshes,
            mesh_name_to_index,
            instances,
            constant_media,
            constant_medium_name_to_index,
        })
    }
}
