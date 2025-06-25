use core::fmt;
use std::collections::{HashMap, hash_map::Entry};

use crate::{MAT_PROP_VALUE_TYPE_CHECKER, Texture, shaders::closest_hit};

#[derive(Debug)]
pub struct CheckerTexture {
    pub scale: f32,
    pub odd: String,
    pub even: String,
}

pub struct CheckerTextures {
    pub textures: Vec<CheckerTexture>,
    pub indices: HashMap<String, u32>,
}

impl CheckerTextures {
    /// Loads all unique checker textures from scene file.
    pub fn new(all_textures: &HashMap<String, Texture>) -> CheckerTextures {
        let mut textures = vec![];
        let mut indices = HashMap::new();

        for texture in all_textures.values() {
            if let Texture::Checker {
                name,
                scale,
                odd,
                even,
            } = texture
            {
                if let Entry::Vacant(e) = indices.entry(name.clone()) {
                    e.insert(textures.len() as u32);

                    textures.push(CheckerTexture {
                        scale: *scale,
                        odd: odd.clone(),
                        even: even.clone(),
                    });
                }
            }
        }

        CheckerTextures { textures, indices }
    }

    pub fn to_shader(&self, name: &str) -> Option<closest_hit::MaterialPropertyValue> {
        self.indices
            .get(name)
            .map(|i| closest_hit::MaterialPropertyValue {
                propValueType: MAT_PROP_VALUE_TYPE_CHECKER,
                index: *i,
            })
    }
}

impl fmt::Debug for CheckerTextures {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CheckerTextures")
            .field("textures", &self.textures)
            .field("indices", &self.indices)
            .finish()
    }
}
