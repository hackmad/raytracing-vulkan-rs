use std::{collections::HashMap, fmt, path::PathBuf};

use egui_winit_vulkano::egui::emath::OrderedFloat;

use super::{Model, shaders::closest_hit};

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
    RGB = 1,
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

    /// Index into material color or texture buffers. -1 => None.
    /// This is done to reduce the overhead of storing both a color value and texture index in
    /// the shader's `Material` struct.
    pub index: i32,
}

impl MaterialPropertyData {
    /// Create material using a solid color.
    fn new_color(prop_type: MaterialPropertyType, index: i32) -> Self {
        Self {
            prop_type: prop_type as _,
            prop_value_type: MaterialPropertyValueType::RGB as _,
            index,
        }
    }

    /// Create material using a texture for sampling colors.
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
        material_color_indices: &HashMap<RgbColor, i32>,
    ) -> Self {
        match value {
            MaterialPropertyValue::None => Self::new_none(prop_type),

            MaterialPropertyValue::RGB { color } => {
                let index = material_color_indices
                    .get(&color.into())
                    .expect(format!("Material color {color:?} not found").as_ref());
                Self::new_color(prop_type, *index)
            }

            MaterialPropertyValue::Texture { path } => {
                let index = texture_indices
                    .get(path)
                    .expect(format!("Texture {path} not found").as_ref());
                Self::new_texture_index(prop_type, *index)
            }
        }
    }
}

impl Into<closest_hit::Material> for MaterialPropertyData {
    fn into(self) -> closest_hit::Material {
        // Convert to the shader's `Material` struct.
        closest_hit::Material {
            propType: self.prop_type,
            propValueType: self.prop_value_type,
            index: self.index,
        }
    }
}

/// Enumerates material property values.
#[derive(Clone, Debug)]
pub enum MaterialPropertyValue {
    /// No value.
    None,

    /// Solid RGB color.
    RGB { color: [f32; 3] },

    /// Texture image path.
    Texture { path: String },
}

impl MaterialPropertyValue {
    /// Create a new material property value.
    pub fn new(
        color: &Option<[f32; 3]>,
        texture: &Option<String>,
        mut parent_path: PathBuf,
    ) -> Self {
        match color {
            Some(c) => MaterialPropertyValue::RGB { color: c.clone() },

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
                        println!("Invalid texture path {path}.");
                        Self::None
                    }
                }
            }),
        }
    }
}

/// Stores unique material RGB values which will be added to to a storage buffer used by the
/// shader.
pub struct MaterialColors {
    /// The material colors.
    pub colors: Vec<[f32; 3]>,

    /// Maps unique colors to their index in `colors`.
    pub indices: HashMap<RgbColor, i32>, /* GLSL int => i32*/
}

impl fmt::Debug for MaterialColors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MaterialColors")
            .field("colors", &self.colors.len())
            .field("indices", &self.indices)
            .finish()
    }
}

impl MaterialColors {
    /// Load all unique texture paths from all models. Assumes images have alpha channel.
    pub fn load(models: &[Model]) -> Self {
        let mut colors = vec![];
        let mut indices = HashMap::new();

        for model in models.iter() {
            if let Some(material) = &model.material {
                match material.diffuse {
                    MaterialPropertyValue::RGB { color } => {
                        let rgb = RgbColor::from(color);
                        if !indices.contains_key(&rgb) {
                            indices.insert(rgb, colors.len() as i32);
                            colors.push(color.clone());
                        }
                    }
                    _ => {}
                }
            }
        }

        Self { colors, indices }
    }
}

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub struct RgbColor {
    pub r: OrderedFloat<f32>,
    pub g: OrderedFloat<f32>,
    pub b: OrderedFloat<f32>,
}

impl fmt::Debug for RgbColor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RgbColor")
            .field("r", &self.r.0)
            .field("g", &self.g.0)
            .field("b", &self.b.0)
            .finish()
    }
}

impl From<[f32; 3]> for RgbColor {
    fn from(value: [f32; 3]) -> Self {
        Self {
            r: value[0].into(),
            g: value[1].into(),
            b: value[2].into(),
        }
    }
}

impl From<&[f32; 3]> for RgbColor {
    fn from(value: &[f32; 3]) -> Self {
        Self {
            r: value[0].into(),
            g: value[1].into(),
            b: value[2].into(),
        }
    }
}

impl Into<[f32; 3]> for RgbColor {
    fn into(self) -> [f32; 3] {
        [self.r.0, self.g.0, self.b.0]
    }
}
