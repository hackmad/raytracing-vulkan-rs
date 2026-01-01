use glam::Mat4;

#[derive(Debug)]
pub enum Transform {
    Static(Mat4),
    Animated { start: Mat4, end: Mat4 },
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
    pub fn get_vulkan_acc_transform(&self) -> [[f32; 4]; 3] {
        match self.object_to_world {
            Transform::Static(mat) => {
                let t = mat.transpose().to_cols_array_2d();
                [t[0], t[1], t[2]]
            }
            Transform::Animated { .. } => todo!("Animated transforms not implemented yet!"),
        }
    }
}
