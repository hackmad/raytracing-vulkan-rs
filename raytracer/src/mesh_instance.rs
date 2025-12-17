use std::sync::Arc;

use crate::Mesh;

#[derive(Debug)]
pub struct MeshInstance {
    pub mesh: Arc<Mesh>,
    pub transform: [[f32; 4]; 3],
}

impl MeshInstance {
    pub fn new(mesh: Arc<Mesh>, transform: &[[f32; 3]; 3]) -> Self {
        Self {
            mesh,
            transform: [
                [transform[0][0], transform[0][1], transform[0][2], 0.0],
                [transform[1][0], transform[1][1], transform[1][2], 0.0],
                [transform[2][0], transform[2][1], transform[2][2], 0.0],
            ],
        }
    }
}
