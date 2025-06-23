use std::{collections::HashMap, path::Path};

use anyhow::{Result, anyhow};
use log::debug;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TextureType {
    Constant {
        name: String,
        rgb: [f32; 3],
    },
    Image {
        name: String,
        path: String,
    },
    Checker {
        name: String,
        scale: f32,
        even: String,
        odd: String,
    },
    Noise {
        name: String,
        scale: f32,
    },
}

impl TextureType {
    pub fn get_name(&self) -> &str {
        match self {
            Self::Constant { name, .. } => name,
            Self::Image { name, .. } => name,
            Self::Checker { name, .. } => name,
            Self::Noise { name, .. } => name,
        }
    }

    pub fn adjust_relative_path(&mut self, relative_to: &Path) {
        if let Self::Image { path, .. } = self {
            let path_buf = Path::new(path).to_path_buf();
            if path_buf.is_relative() {
                let mut new_path_buf = relative_to.to_path_buf();
                new_path_buf.push(path_buf);
                *path = new_path_buf.to_str().unwrap().to_owned();
            }
        }
    }

    pub fn is_valid(&self, all_textures: &HashMap<String, Self>) -> Result<()> {
        match self {
            Self::Constant { .. } | Self::Image { .. } | Self::Noise { .. } => Ok(()),
            Self::Checker {
                name, odd, even, ..
            } => match all_textures.get(odd) {
                Some(Self::Constant { .. })
                | Some(Self::Image { .. })
                | Some(Self::Noise { .. }) => Ok(()),
                Some(Self::Checker { .. }) => Err(anyhow!("Checker texture cannot be recursive.")),
                None => Err(anyhow!(
                    "Check texture {name} references unknown texture odd={odd}"
                )),
            }
            .and(match all_textures.get(even) {
                Some(Self::Constant { .. })
                | Some(Self::Image { .. })
                | Some(Self::Noise { .. }) => Ok(()),
                Some(Self::Checker { .. }) => Err(anyhow!("Checker texture cannot be recursive.")),
                None => Err(anyhow!(
                    "Check texture {name} references unknown texture even={even}"
                )),
            }),
        }
    }

    /// This will find cycles where any texture could refer to another texture like
    /// TextureType::Checker.
    ///
    /// For now this is not used.
    #[allow(unused)]
    pub fn find_cycles(&self, all_textures: &HashMap<String, Self>) -> Option<String> {
        let mut seen = vec![];
        if self.find_cycles_internal(all_textures, &mut seen) {
            Some(seen.join(", "))
        } else {
            None
        }
    }

    fn find_cycles_internal(
        &self,
        all_textures: &HashMap<String, Self>,
        seen: &mut Vec<String>,
    ) -> bool {
        debug!("{seen:?}");
        match self {
            Self::Constant { name, .. } => seen.contains(name),
            Self::Image { name, .. } => seen.contains(name),
            Self::Noise { name, .. } => seen.contains(name),
            Self::Checker {
                name, even, odd, ..
            } => {
                if seen.contains(name) {
                    return true;
                }
                seen.push(name.clone());

                if let Some(tex) = all_textures.get(odd) {
                    if tex.find_cycles_internal(all_textures, seen) {
                        return true;
                    }
                }

                if let Some(tex) = all_textures.get(even) {
                    if tex.find_cycles_internal(all_textures, seen) {
                        return true;
                    }
                }

                seen.pop();
                false
            }
        }
    }
}
