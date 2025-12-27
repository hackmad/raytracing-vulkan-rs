use glam::Mat4;

#[derive(Debug)]
pub struct MeshInstance {
    pub mesh_index: usize,
    pub object_to_world_space_matrix: Mat4,
}

impl MeshInstance {
    pub fn new(mesh_index: usize, object_to_world_space_matrix: Mat4) -> Self {
        Self {
            mesh_index,
            object_to_world_space_matrix,
        }
    }

    /// Returns the 4x3 matrix used in Vulkan transformations for acceleration structures.
    pub fn get_vulkan_acc_transform(&self) -> [[f32; 4]; 3] {
        let t = self
            .object_to_world_space_matrix
            .transpose()
            .to_cols_array_2d();
        [t[0], t[1], t[2]]
    }
}
