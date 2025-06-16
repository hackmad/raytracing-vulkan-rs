use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Render {
    pub camera: String,
    pub samples_per_pixel: u32, // See ray_gen.glsl. Don't exceed 64.
    pub sample_batches: u32,    // See ray_gen.glsl. Don't exceed 32.
    pub max_ray_depth: u32,
}
