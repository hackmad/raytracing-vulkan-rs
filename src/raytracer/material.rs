use super::{Model, shaders::closest_hit};
use log::error;
use ordered_float::OrderedFloat;
use std::{collections::HashMap, collections::hash_map::Entry, fmt, path::PathBuf};

/// Material property types. These will correspond to `MAT_PROP_TYPE_*` constants in the shader source.
#[derive(Clone, Copy, Debug)]
#[repr(u32)]
pub enum MaterialPropertyType {
    Diffuse = 0,
}

/// Material property value types. These will correspond to `MAT_PROP_VALUE_TYPE_*` constants in the shader source.
#[derive(Clone, Copy, Debug)]
#[repr(u32)]
pub enum MaterialPropertyValueType {
    None = 0,
    Rgb = 1,
    Texture = 2,
}

/// Represents the `Material` struct in shader source.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct MaterialPropertyData {
    /// The `MaterialPropertyType`.
    pub prop_type: u32,

    /// The `MaterialPropertyValueType`.
    pub prop_value_type: u32,

    /// Index into material colour or texture buffers. -1 => None.
    /// This is done to reduce the overhead of storing both a colour value and texture index in
    /// the shader's `Material` struct.
    pub index: i32,
}

impl MaterialPropertyData {
    /// Create material using a solid colour.
    fn new_colour(prop_type: MaterialPropertyType, index: i32) -> Self {
        Self {
            prop_type: prop_type as _,
            prop_value_type: MaterialPropertyValueType::Rgb as _,
            index,
        }
    }

    /// Create material using a texture for sampling colours.
    fn new_texture_index(prop_type: MaterialPropertyType, index: i32) -> Self {
        Self {
            prop_type: prop_type as _,
            prop_value_type: MaterialPropertyValueType::Texture as _,
            index,
        }
    }

    /// Create a material property with no value.
    pub fn new_none(prop_type: MaterialPropertyType) -> Self {
        Self {
            prop_type: prop_type as _,
            prop_value_type: MaterialPropertyValueType::None as _,
            index: -1,
        }
    }

    /// Create material for the given property type and value.
    pub fn from_property_value(
        prop_type: MaterialPropertyType,
        value: &MaterialPropertyValue,
        texture_indices: &HashMap<String, i32>,
        material_colour_indices: &HashMap<RgbColour, i32>,
    ) -> Self {
        match value {
            MaterialPropertyValue::None => Self::new_none(prop_type),

            MaterialPropertyValue::Rgb { colour } => {
                let index = material_colour_indices
                    .get(&colour.into())
                    .unwrap_or_else(|| panic!("Material colour {colour:?} not found"));
                Self::new_colour(prop_type, *index)
            }

            MaterialPropertyValue::Texture { path } => {
                let index = texture_indices
                    .get(path)
                    .unwrap_or_else(|| panic!("Texture {path} not found"));
                Self::new_texture_index(prop_type, *index)
            }
        }
    }
}

impl From<MaterialPropertyData> for closest_hit::MaterialPropertyValue {
    fn from(mat: MaterialPropertyData) -> Self {
        // Convert to the shader's `Material` struct.
        Self {
            propType: mat.prop_type,
            propValueType: mat.prop_value_type,
            index: mat.index,
        }
    }
}

/// Enumerates material property values.
#[derive(Clone, Debug)]
pub enum MaterialPropertyValue {
    /// No value.
    None,

    /// Solid RGB colour.
    Rgb { colour: [f32; 3] },

    /// Texture image path.
    Texture { path: String },
}

impl MaterialPropertyValue {
    /// Create a new material property value.
    pub fn new(
        colour: &Option<[f32; 3]>,
        texture: &Option<String>,
        mut parent_path: PathBuf,
    ) -> Self {
        match colour {
            Some(c) => MaterialPropertyValue::Rgb { colour: *c },

            None => texture.clone().map_or(Self::None, |path| {
                if PathBuf::from(&path).is_absolute() {
                    Self::Texture { path }
                } else {
                    parent_path.push(&path);

                    if let Some(path) = parent_path.to_str() {
                        Self::Texture {
                            path: path.to_string(),
                        }
                    } else {
                        error!("Invalid texture path {path}.");
                        Self::None
                    }
                }
            }),
        }
    }
}

/// Stores unique material RGB values which will be added to to a storage buffer used by the
/// shader.
pub struct MaterialColours {
    /// The material colours.
    pub colours: Vec<[f32; 3]>,

    /// Maps unique colours to their index in `colours`.
    pub indices: HashMap<RgbColour, i32>, /* GLSL int => i32*/
}

impl fmt::Debug for MaterialColours {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MaterialColours")
            .field("colours", &self.colours.len())
            .field("indices", &self.indices)
            .finish()
    }
}

impl MaterialColours {
    /// Load all unique texture paths from all models. Assumes images have alpha channel.
    pub fn load(models: &[Model]) -> Self {
        let mut colours = vec![];
        let mut indices = HashMap::new();

        for model in models.iter() {
            if let Some(material) = &model.material {
                if let MaterialPropertyValue::Rgb { colour } = material.diffuse {
                    let rgb = RgbColour::from(colour);
                    if let Entry::Vacant(e) = indices.entry(rgb) {
                        e.insert(colours.len() as i32);
                        colours.push(colour);
                    }
                }
            }
        }

        Self { colours, indices }
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
