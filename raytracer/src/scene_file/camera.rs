use std::sync::{Arc, RwLock};

use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::PerspectiveCamera;

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

    pub fn to_camera(&self, image_width: u32, image_height: u32) -> Arc<RwLock<dyn crate::Camera>> {
        match self {
            Self::Perspective {
                name: _,
                eye,
                look_at,
                up,
                fov_y,
                z_near,
                z_far,
                focal_length,
                aperture_size,
            } => Arc::new(RwLock::new(PerspectiveCamera::new(
                Vec3::from_slice(eye),
                Vec3::from_slice(look_at),
                Vec3::from_slice(up),
                fov_y.to_radians(),
                *z_near,
                *z_far,
                *focal_length,
                *aperture_size,
                image_width,
                image_height,
            ))),
        }
    }
}
