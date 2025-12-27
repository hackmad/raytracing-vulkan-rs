#[derive(Debug)]
pub struct MeshInstance {
    pub mesh_index: usize,
    pub transform: [[f32; 4]; 3],
}

impl MeshInstance {
    pub fn new(mesh_index: usize, transform: [[f32; 4]; 3]) -> Self {
        Self {
            mesh_index,
            transform,
        }
    }
}
