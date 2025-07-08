// NOTE: Update Materials::to_shader() when adding new materials.
pub const MAT_TYPE_NONE: u32 = 0;
pub const MAT_TYPE_LAMBERTIAN: u32 = 1;
pub const MAT_TYPE_METAL: u32 = 2;
pub const MAT_TYPE_DIELECTRIC: u32 = 3;
pub const MAT_TYPE_DIFFUSE_LIGHT: u32 = 4;

pub const MAT_PROP_VALUE_TYPE_RGB: u32 = 0;
pub const MAT_PROP_VALUE_TYPE_IMAGE: u32 = 1;
pub const MAT_PROP_VALUE_TYPE_CHECKER: u32 = 2;
pub const MAT_PROP_VALUE_TYPE_NOISE: u32 = 3;

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct MaterialPropertyValue {
    pub prop_value_type: u32,
    pub index: u32,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct LambertianMaterial {
    pub albedo: MaterialPropertyValue,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct MetalMaterial {
    pub albedo: MaterialPropertyValue,
    pub fuzz: MaterialPropertyValue,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct DielectricMaterial {
    pub refraction_index: f32,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct DiffuseLightMaterial {
    pub emit: MaterialPropertyValue,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct CheckerTexture {
    pub scale: f32,
    pub odd: MaterialPropertyValue,
    pub even: MaterialPropertyValue,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct NoiseTexture {
    pub scale: f32,
}
