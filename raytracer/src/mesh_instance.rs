use crate::DecomposedTransform;

/// Stores decomposed transformations for static or moving mesh instances.
#[derive(Debug)]
pub enum Transform {
    /// Single transform for static mesh instances.
    Static(DecomposedTransform),

    /// Start and end transforms for moving mesh instances.
    Animated {
        start: DecomposedTransform,
        end: DecomposedTransform,
    },
}

impl From<scene_file::Matrix> for Transform {
    /// Decompose [scene_file::Matrix] to an equivalent [Transform].
    fn from(value: scene_file::Matrix) -> Self {
        match value {
            scene_file::Matrix::Static(mat) => Transform::Static(DecomposedTransform::from(mat)),

            scene_file::Matrix::Animated(mat1, mat2) => Transform::Animated {
                start: DecomposedTransform::from(mat1),
                end: DecomposedTransform::from(mat2),
            },
        }
    }
}

/// Stores mesh instance related data.
#[derive(Debug)]
pub struct MeshInstance {
    /// Index of the mesh.
    pub mesh_index: usize,

    /// Transformation for this instance.
    pub object_to_world: Transform,
}

impl MeshInstance {
    /// Create a new mesh instance with a given mesh index and object-to-world transformation.
    pub fn new(mesh_index: usize, object_to_world: Transform) -> Self {
        Self {
            mesh_index,
            object_to_world,
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
