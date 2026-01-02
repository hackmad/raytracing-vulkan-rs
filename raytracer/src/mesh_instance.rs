use crate::AnimatedTransform;

#[derive(Debug)]
pub enum Transform {
    Static(AnimatedTransform),
    Animated {
        start: AnimatedTransform,
        end: AnimatedTransform,
    },
}

#[derive(Debug)]
pub struct MeshInstance {
    pub mesh_index: usize,
    pub object_to_world: Transform,
}

impl MeshInstance {
    pub fn new(mesh_index: usize, object_to_world: Transform) -> Self {
        Self {
            mesh_index,
            object_to_world,
        }
    }

    /// Returns the 4x3 matrix used in Vulkan transformations for acceleration structures.
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
