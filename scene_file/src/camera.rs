use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Camera {
    Perspective {
        name: String,
        eye: [f32; 3],
        look_at: [f32; 3],
        up: [f32; 3],
        fov_y: f32, // Vertical FOV in degrees.
        z_near: f32,
        z_far: f32,
        focal_length: f32,
        aperture_size: f32,
    },
}

impl Camera {
    pub fn get_name(&self) -> &str {
        match self {
            Self::Perspective { name, .. } => name,
        }
    }
}
