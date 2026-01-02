use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Instance {
    pub name: String,
    pub transform: Option<TransformType>,
}

impl Instance {
    /// Returns final transformation matrix/matrices for the object in world space.
    pub fn get_object_to_world_space_matrix(&self) -> Matrix {
        self.transform
            .as_ref()
            .map_or(Matrix::Static(Mat4::IDENTITY), |t| t.into())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TransformType {
    Static(Transform),
    Animated(Transform, Transform),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Rotate {
    pub axis: [f32; 3],
    pub degrees: f32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Transform {
    pub translate: Option<[f32; 3]>,
    pub rotate: Option<Rotate>,
    pub scale: Option<[f32; 3]>,
}

impl Transform {
    pub fn to_matrix(&self) -> Mat4 {
        let t = self.translate.as_ref().map_or(Mat4::IDENTITY, |d| {
            Mat4::from_translation(Vec3::new(d[0], d[1], d[2]))
        });
        let r = self.rotate.as_ref().map_or(Mat4::IDENTITY, |r| {
            Mat4::from_axis_angle(r.axis.into(), r.degrees.to_radians())
        });
        let s = self.scale.as_ref().map_or(Mat4::IDENTITY, |s| {
            Mat4::from_scale(Vec3::new(s[0], s[1], s[2]))
        });
        t.mul_mat4(&r).mul_mat4(&s)
    }
}

#[derive(Debug)]
pub enum Matrix {
    Static(Mat4),
    Animated(Mat4, Mat4),
}
impl From<&TransformType> for Matrix {
    fn from(value: &TransformType) -> Self {
        match value {
            TransformType::Static(t) => Self::Static(t.to_matrix()),
            TransformType::Animated(start, end) => {
                Self::Animated(start.to_matrix(), end.to_matrix())
            }
        }
    }
}
