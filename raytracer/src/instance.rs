use std::{collections::HashMap, sync::Arc};

use anyhow::{Result, anyhow};
use scene_file::{InstanceType, SceneFile};

use crate::{ConstantMedium, MediumType, Mesh, Transform, Volume};

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
    pub volumes: Vec<Arc<Volume>>,
    pub mesh_name_to_index: HashMap<String, usize>,
    pub volume_name_to_index: HashMap<String, usize>,
    pub mesh_instances: Vec<Instance>,
    pub volume_instances: Vec<Instance>,
    pub constant_media: Vec<ConstantMedium>,
}

impl Instances {
    pub fn new(scene_file: &SceneFile) -> Result<Self> {
        let mut meshes: Vec<Arc<Mesh>> = Vec::new();
        let mut volumes: Vec<Arc<Volume>> = Vec::new();
        let mut mesh_name_to_index: HashMap<String, usize> = HashMap::new();
        let mut volume_name_to_index: HashMap<String, usize> = HashMap::new();
        let mut constant_media: Vec<ConstantMedium> = Vec::new();

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

            match instance.instance_type {
                InstanceType::Surface => {
                    let mesh = Arc::new(Mesh::from(primitive));
                    mesh_name_to_index.insert(instance.name.clone(), meshes.len());
                    meshes.push(mesh);
                }

                InstanceType::ConstantMedium { density } => {
                    constant_media.push(ConstantMedium::new(density));
                    let medium_index = constant_media.len() - 1;

                    if let Some(volume) =
                        Volume::from_primitive(primitive, MediumType::ConstantMedium, medium_index)
                    {
                        let volume = Arc::new(volume);
                        volume_name_to_index.insert(primitive.get_name().into(), volumes.len());
                        volumes.push(volume);
                    } else {
                        return Err(anyhow!(
                            "Constant medium instance creation not supported for primitive {}",
                            primitive.get_name()
                        ));
                    }
                }
            }
        }

        // Get instances.
        let mut mesh_instances: Vec<Instance> = Vec::new();
        let mut volume_instances: Vec<Instance> = Vec::new();
        for instance in scene_file.instances.iter() {
            let object_to_world = instance.get_object_to_world_space_matrix();
            let transform = Transform::from(object_to_world);

            match instance.instance_type {
                InstanceType::Surface => {
                    if let Some(index) = mesh_name_to_index.get(&instance.name) {
                        mesh_instances.push(Instance::new(
                            *index,
                            InstanceType::Surface,
                            transform,
                        ));
                    } else {
                        return Err(anyhow!("Mesh {} not found", instance.name));
                    }
                }

                InstanceType::ConstantMedium { density } => {
                    if let Some(index) = volume_name_to_index.get(&instance.name) {
                        volume_instances.push(Instance::new(
                            *index,
                            InstanceType::ConstantMedium { density },
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
            volumes,
            mesh_name_to_index,
            volume_name_to_index,
            mesh_instances,
            volume_instances,
            constant_media,
        })
    }
}
