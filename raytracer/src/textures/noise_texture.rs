use core::fmt;
use std::collections::{HashMap, hash_map::Entry};

use scene_file::Texture;
use shaders::closest_hit;

use crate::MAT_PROP_VALUE_TYPE_NOISE;

#[derive(Debug)]
pub struct NoiseTexture {
    pub scale: f32,
}

pub struct NoiseTextures {
    pub textures: Vec<NoiseTexture>,
    pub indices: HashMap<String, u32>,
}

impl NoiseTextures {
    pub fn new(all_textures: &HashMap<String, Texture>) -> Self {
        let mut textures = vec![];
        let mut indices = HashMap::new();

        for texture in all_textures.values() {
            if let Texture::Noise { name, scale } = texture {
                if let Entry::Vacant(e) = indices.entry(name.clone()) {
                    e.insert(textures.len() as u32);
                    textures.push(NoiseTexture { scale: *scale });
                }
            }
        }

        Self { textures, indices }
    }

    pub fn to_shader(&self, name: &str) -> Option<closest_hit::MaterialPropertyValue> {
        self.indices
            .get(name)
            .map(|i| closest_hit::MaterialPropertyValue {
                propValueType: MAT_PROP_VALUE_TYPE_NOISE,
                index: *i,
            })
    }
}

impl fmt::Debug for NoiseTextures {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NoiseTextures")
            .field("textures", &self.textures)
            .field("indices", &self.indices)
            .finish()
    }
}
