#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct RayGenPushConstants {
    pub resolution: [u32; 2],
    pub samples_per_pixel: u32,
    pub sample_batches: u32,
    pub sample_batch: u32,
    pub max_ray_depth: u32,
}

impl RayGenPushConstants {
    pub fn to_raw_bytes(&self) -> &[u8] {
        // SAFETY: We are converting a plain-old-data struct to a &[u8] slice
        unsafe {
            std::slice::from_raw_parts(
                (self as *const RayGenPushConstants) as *const u8,
                std::mem::size_of::<RayGenPushConstants>(),
            )
        }
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct ClosestHitPushConstants {
    pub mesh_count: u32,
    pub image_texture_count: u32,
    pub constant_colour_count: u32,
    pub checker_texture_count: u32,
    pub noise_texture_count: u32,
    pub lambertian_material_count: u32,
    pub metal_material_count: u32,
    pub dielectric_material_count: u32,
    pub diffuse_light_material_count: u32,
}

impl ClosestHitPushConstants {
    pub fn to_raw_bytes(&self) -> &[u8] {
        // SAFETY: We are converting a plain-old-data struct to a &[u8] slice
        unsafe {
            std::slice::from_raw_parts(
                (self as *const ClosestHitPushConstants) as *const u8,
                std::mem::size_of::<ClosestHitPushConstants>(),
            )
        }
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct UnifiedPushConstants {
    // RayGen: 0–23
    pub ray_gen_pc: RayGenPushConstants,

    // ClosestHit: 24–55
    pub closest_hit_pc: ClosestHitPushConstants,
}
