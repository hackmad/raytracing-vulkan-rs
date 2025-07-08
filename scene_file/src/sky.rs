use serde::{Deserialize, Serialize};

const _SKY_TYPE_NONE: u32 = 0;
const SKY_TYPE_SOLID: u32 = 1;
const SKY_TYPE_VERTICAL_GRADIENT: u32 = 2;

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Sky {
    Solid {
        rgb: [f32; 3],
    },
    VerticalGradient {
        factor: f32,
        top: [f32; 3],
        bottom: [f32; 3],
    },
}

impl Sky {
    pub fn to_shader(&self) -> shaders::Sky {
        match self {
            Self::Solid { rgb } => shaders::Sky {
                sky_type: SKY_TYPE_SOLID,
                solid: *rgb,
                v_factor: 0.0,
                v_top: *rgb,
                v_bottom: *rgb,
            },
            Self::VerticalGradient {
                factor,
                top,
                bottom,
            } => shaders::Sky {
                sky_type: SKY_TYPE_VERTICAL_GRADIENT,
                solid: *top,
                v_factor: *factor,
                v_top: *top,
                v_bottom: *bottom,
            },
        }
    }
}
