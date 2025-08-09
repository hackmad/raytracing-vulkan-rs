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

    _padding: u32,
}

impl Sky {
    pub fn none() -> Self {
        Self {
            sky_type: SKY_TYPE_NONE,
            solid: [0.0, 0.0, 0.0],
            v_top: [0.0, 0.0, 0.0],
            v_factor: 0.0,
            v_bottom: [0.0, 0.0, 0.0],
            _padding: 0,
        }
    }

    pub fn solid(solid: [f32; 3]) -> Self {
        Self {
            sky_type: SKY_TYPE_SOLID,
            solid,
            v_top: [0.0, 0.0, 0.0],
            v_factor: 0.0,
            v_bottom: [0.0, 0.0, 0.0],
            _padding: 0,
        }
    }

    pub fn vertical_gradient(factor: f32, top: [f32; 3], bottom: [f32; 3]) -> Self {
        Self {
            sky_type: SKY_TYPE_VERTICAL_GRADIENT,
            v_factor: factor,
            v_top: top,
            v_bottom: bottom,
            solid: [0.0, 0.0, 0.0],
            _padding: 0,
        }
    }
}
