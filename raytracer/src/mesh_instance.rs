use std::sync::Arc;

use crate::Mesh;

#[derive(Debug)]
pub struct MeshInstance {
    pub mesh: Arc<Mesh>,
    pub transform: [[f32; 4]; 3],
}

impl MeshInstance {
    pub fn new(mesh: Arc<Mesh>, transform: [[f32; 4]; 3]) -> Self {
        Self { mesh, transform }
    }
}
