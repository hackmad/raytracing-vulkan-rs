#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Camera {
    pub view_proj: [[f32; 4]; 4],    // 64 bytes
    pub view_inverse: [[f32; 4]; 4], // 64 bytes
    pub proj_inverse: [[f32; 4]; 4], // 64 bytes
    pub focal_length: f32,           // 4 bytes
    pub aperture_size: f32,          // 4 bytes
    _padding: [f32; 2],              // 8 bytes padding to align to 16 bytes
}

impl Camera {
    pub fn new(
        view_proj: [[f32; 4]; 4],
        view_inverse: [[f32; 4]; 4],
        proj_inverse: [[f32; 4]; 4],
        focal_length: f32,
        aperture_size: f32,
    ) -> Self {
        Self {
            view_proj,
            view_inverse,
            proj_inverse,
            focal_length,
            aperture_size,
            _padding: [0.0, 0.0],
        }
    }
}
