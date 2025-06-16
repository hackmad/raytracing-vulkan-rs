use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectType {
    UvSphere {
        name: String,
        center: [f32; 3],
        radius: f32,
        rings: u32,
        segments: u32,
        material: String,
    },
}
