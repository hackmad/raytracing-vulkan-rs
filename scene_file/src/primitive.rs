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

    pub fn get_aabb(&self) -> [[f32; 3]; 2] {
        match self {
            Self::UvSphere {
                center: c,
                radius: r,
                ..
            } => [
                [c[0] - *r, c[1] - *r, c[2] - *r],
                [c[0] + *r, c[1] + *r, c[2] + *r],
            ],
            Self::Triangle { points, .. } => get_aabb_points(points),
            Self::Quad { points, .. } => get_aabb_points(points),
            Self::Box { corners, .. } => get_aabb_points(corners),
        }
    }
}

fn get_aabb_points(points: &[[f32; 3]]) -> [[f32; 3]; 2] {
    let mut min = points[0];
    let mut max = points[0];

    for p in points.iter().skip(1) {
        for i in 0..3 {
            min[i] = min[i].min(p[i]);
            max[i] = max[i].max(p[i]);
        }
    }

    [min, max]
}
