#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Camera {
    pub view_proj: [[f32; 4]; 4],
    pub view_inverse: [[f32; 4]; 4],
    pub proj_inverse: [[f32; 4]; 4],
    pub focal_length: f32,
    pub aperture_size: f32,
}
