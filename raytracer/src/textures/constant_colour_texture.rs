use core::fmt;
use std::collections::{HashMap, hash_map::Entry};

use ordered_float::OrderedFloat;
use scene_file::Texture;
use shaders::closest_hit;

use crate::MAT_PROP_VALUE_TYPE_RGB;

/// Stores unique material RGB values which will be added to to a storage buffer used by the
/// shader.
pub struct ConstantColourTextures {
    /// The material colours. This will be used to create the storage buffers for shaders.
    pub colours: Vec<[f32; 3]>,

    /// Maps unique colours to their index in `colours`. These indices are used in the
    /// MaterialPropertyValue structure.
    pub indices: HashMap<String, u32>,
}

impl ConstantColourTextures {
    /// Loads all unique colours from scene file.
    pub fn new(textures: &HashMap<String, Texture>) -> ConstantColourTextures {
        let mut colours = vec![];
        let mut indices = HashMap::new();

        for texture in textures.values() {
            if let Texture::Constant { name, rgb } = texture {
                if let Entry::Vacant(e) = indices.entry(name.clone()) {
                    e.insert(colours.len() as _);
                    colours.push(*rgb);
                }
            }
        }

        ConstantColourTextures { colours, indices }
    }

    pub fn to_shader(&self, name: &str) -> Option<closest_hit::MaterialPropertyValue> {
        self.indices
            .get(name)
            .map(|i| closest_hit::MaterialPropertyValue {
                propValueType: MAT_PROP_VALUE_TYPE_RGB,
                index: *i,
            })
    }
}

impl fmt::Debug for ConstantColourTextures {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConstantColourTextures")
            .field("colours", &self.colours)
            .field("indices", &self.indices)
            .finish()
    }
}

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub struct RgbColour {
    pub r: OrderedFloat<f32>,
    pub g: OrderedFloat<f32>,
    pub b: OrderedFloat<f32>,
}

impl fmt::Debug for RgbColour {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RgbColour")
            .field("r", &self.r.0)
            .field("g", &self.g.0)
            .field("b", &self.b.0)
            .finish()
    }
}

impl From<[f32; 3]> for RgbColour {
    fn from(value: [f32; 3]) -> Self {
        Self {
            r: value[0].into(),
            g: value[1].into(),
            b: value[2].into(),
        }
    }
}

impl From<&[f32; 3]> for RgbColour {
    fn from(value: &[f32; 3]) -> Self {
        Self {
            r: value[0].into(),
            g: value[1].into(),
            b: value[2].into(),
        }
    }
}

impl From<RgbColour> for [f32; 3] {
    fn from(c: RgbColour) -> Self {
        [c.r.0, c.g.0, c.b.0]
    }
}
