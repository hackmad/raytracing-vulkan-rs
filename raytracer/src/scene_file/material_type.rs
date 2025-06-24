use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MaterialType {
    Lambertian {
        name: String,
        albedo: String,
    },
    Metal {
        name: String,
        albedo: String,
        fuzz: String,
    },
    Dielectric {
        name: String,
        refraction_index: f32,
    },
    DiffuseLight {
        name: String,
        emit: String,
    },
}

impl MaterialType {
    pub fn get_name(&self) -> &str {
        match self {
            Self::Lambertian { name, .. } => name.as_ref(),
            Self::Metal { name, .. } => name.as_ref(),
            Self::Dielectric { name, .. } => name.as_ref(),
            Self::DiffuseLight { name, .. } => name.as_ref(),
        }
    }
}
