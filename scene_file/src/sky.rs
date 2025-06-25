use serde::{Deserialize, Serialize};
use shaders::ray_gen;

const _SKY_TYPE_NONE: u32 = 0;
const SKY_TYPE_SOLID: u32 = 1;
const SKY_TYPE_VERTICAL_GRADIENT: u32 = 2;

#[derive(Clone, Debug, Deserialize, Serialize)]
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
    pub fn to_shader(&self) -> ray_gen::Sky {
        match self {
            Self::Solid { rgb } => ray_gen::Sky {
                skyType: SKY_TYPE_SOLID,
                solid: *rgb,
                vFactor: 0.0,
                vTop: *rgb,
                vBottom: *rgb,
            },
            Self::VerticalGradient {
                factor,
                top,
                bottom,
            } => ray_gen::Sky {
                skyType: SKY_TYPE_VERTICAL_GRADIENT,
                solid: *top,
                vFactor: *factor,
                vTop: *top,
                vBottom: *bottom,
            },
        }
    }
}
