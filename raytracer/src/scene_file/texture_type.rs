use std::path::Path;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TextureType {
    Constant { name: String, rgb: [f32; 3] },
    Image { name: String, path: String },
}

impl TextureType {
    pub fn get_name(&self) -> &str {
        match self {
            Self::Constant { name, .. } => name.as_ref(),
            Self::Image { name, .. } => name.as_ref(),
        }
    }

    pub fn adjust_relative_path(&mut self, relative_to: &Path) {
        match self {
            Self::Image { path, .. } => {
                let path_buf = Path::new(path).to_path_buf();
                if path_buf.is_relative() {
                    let mut new_path_buf = relative_to.to_path_buf();
                    new_path_buf.push(path_buf);
                    *path = new_path_buf.to_str().unwrap().to_owned();
                }
            }
            Self::Constant { .. } => {}
        }
    }
}
