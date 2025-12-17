use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct Instance {
    pub name: String,
    pub transform: Option<[[f32; 3]; 3]>,
}
