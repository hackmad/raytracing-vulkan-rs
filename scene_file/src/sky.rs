use serde::{Deserialize, Serialize};

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
            Self::Solid { rgb } => shaders::Sky::solid(*rgb),
            Self::VerticalGradient {
                factor,
                top,
                bottom,
            } => shaders::Sky::vertical_gradient(*factor, *top, *bottom),
        }
    }
}
