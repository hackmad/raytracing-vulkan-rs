use super::shaders::closest_hit;

/// Light property types. These will correspond to `LIGHT_PROP_TYPE_*` constants in the shader source.
#[derive(Clone, Copy, Debug)]
#[repr(u32)]
pub enum LightPropertyType {
    Position = 0,
    Directional = 1,
}

/// Represents the `Light` struct in shader source.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct LightPropertyData {
    /// The `LightPropertyType`.
    pub prop_type: u32,

    /// The intensity.
    intensity: f32,

    /// Location of spot light or vector for directional light source.
    position_or_direction: [f32; 3],
}

impl LightPropertyData {
    /// Create spot light.
    pub fn new_spot(intensity: f32, position: [f32; 3]) -> Self {
        Self {
            prop_type: LightPropertyType::Position as _,
            intensity,
            position_or_direction: position,
        }
    }

    /// Create directional.
    pub fn new_directional(intensity: f32, direction: [f32; 3]) -> Self {
        Self {
            prop_type: LightPropertyType::Directional as _,
            intensity,
            position_or_direction: direction,
        }
    }
}

impl Into<closest_hit::Light> for &LightPropertyData {
    fn into(self) -> closest_hit::Light {
        // Convert to the shader's `Light` struct.
        closest_hit::Light {
            propType: self.prop_type,
            intensity: self.intensity.into(),
            positionOrDirection: self.position_or_direction,
        }
    }
}
