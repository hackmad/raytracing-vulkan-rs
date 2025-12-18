use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Primitive {
    UvSphere {
        name: String,
        center: [f32; 3],
        radius: f32,
        rings: u32,
        segments: u32,
        material: String,
    },
    Triangle {
        name: String,
        points: [[f32; 3]; 3],
        normal: [f32; 3],
        uv: [[f32; 2]; 3],
        material: String,
    },
    Quad {
        name: String,
        points: [[f32; 3]; 4],
        normal: [f32; 3],
        uv: [[f32; 2]; 4],
        material: String,
    },
    Box {
        name: String,
        corners: [[f32; 3]; 2],
        material: String,
    },
}

impl Primitive {
    pub fn get_name(&self) -> &str {
        match self {
            Self::UvSphere { name, .. } => name,
            Self::Triangle { name, .. } => name,
            Self::Quad { name, .. } => name,
            Self::Box { name, .. } => name,
        }
    }
}
