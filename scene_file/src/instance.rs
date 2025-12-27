use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Instance {
    pub name: String,
    pub transforms: Option<Vec<Transform>>,
}

impl Instance {
    /// Multiplies all transformation matrices in order to get final transformation for the
    /// object in world space.
    pub fn get_object_to_world_space_matrix(&self) -> Mat4 {
        match &self.transforms {
            None => Mat4::IDENTITY,
            Some(transforms) => transforms.iter().fold(Mat4::IDENTITY, |acc, transform| {
                acc.mul_mat4(&transform.to_matrix())
            }),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Transform {
    Translate([f32; 3]),
    RotateX(f32),
    RotateY(f32),
    RotateZ(f32),
    Scale([f32; 3]),
}

impl Transform {
    pub fn to_matrix(&self) -> Mat4 {
        match self {
            Self::Translate(d) => Mat4::from_translation(Vec3::new(d[0], d[1], d[2])),
            Self::RotateX(degrees) => {
                Mat4::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), degrees.to_radians())
            }
            Self::RotateY(degrees) => {
                Mat4::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), degrees.to_radians())
            }
            Self::RotateZ(degrees) => {
                Mat4::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), degrees.to_radians())
            }
            Self::Scale(s) => Mat4::from_scale(Vec3::new(s[0], s[1], s[2])),
        }
    }
}
