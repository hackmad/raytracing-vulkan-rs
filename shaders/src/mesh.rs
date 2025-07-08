#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct MeshVertex {
    pub p: [f32; 3], // position
    pub u: f32,      // u- texture coordinate
    pub n: [f32; 3], // normal
    pub v: f32,      // v- texture coordinate
}

impl MeshVertex {
    pub fn new(p: [f32; 3], n: [f32; 3], uv: [f32; 2]) -> Self {
        Self {
            p,
            n,
            u: uv[0],
            v: uv[1],
        }
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Mesh {
    pub vertex_buffer_size: u32,
    pub index_buffer_size: u32,
    pub material_type: u32,
    pub material_index: u32,
}
