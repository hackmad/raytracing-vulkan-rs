pub const SKY_TYPE_NONE: u32 = 0;
pub const SKY_TYPE_SOLID: u32 = 1;
pub const SKY_TYPE_VERTICAL_GRADIENT: u32 = 2;

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Sky {
    /// Solid colour.
    pub solid: [f32; 3],

    /// Sky type.
    pub sky_type: u32,

    /// Vertical gradient
    pub v_top: [f32; 3],
    pub v_factor: f32,
    pub v_bottom: [f32; 3],
}
