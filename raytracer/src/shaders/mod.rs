use std::{fmt, sync::Arc};

use vulkano::{
    device::Device,
    pipeline::{PipelineShaderStageCreateInfo, ray_tracing::RayTracingShaderGroupCreateInfo},
};

pub mod ray_gen {
    vulkano_shaders::shader! {
        ty: "raygen",
        path: "src/shaders/ray_gen.glsl",
        vulkan_version: "1.3",
    }
}

pub mod closest_hit {
    vulkano_shaders::shader! {
        ty: "closesthit",
        path: "src/shaders/closest_hit.glsl",
        vulkan_version: "1.3",
    }
}

pub mod ray_miss {
    vulkano_shaders::shader! {
        ty: "miss",
        path: "src/shaders/ray_miss.glsl",
        vulkan_version: "1.3",
    }
}

pub struct ShaderModules {
    pub stages: Vec<PipelineShaderStageCreateInfo>,
    pub groups: Vec<RayTracingShaderGroupCreateInfo>,
}

impl ShaderModules {
    pub fn load(device: Arc<Device>) -> Self {
        let ray_gen = ray_gen::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();

        let closest_hit = closest_hit::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();

        let ray_miss = ray_miss::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();

        // Make a list of the shader stages that the pipeline will have.
        let stages = vec![
            PipelineShaderStageCreateInfo::new(ray_gen),
            PipelineShaderStageCreateInfo::new(ray_miss),
            PipelineShaderStageCreateInfo::new(closest_hit),
        ];

        // Define the shader groups that will eventually turn into the shader binding table.
        // The numbers are the indices of the stages in the `stages` array.
        let groups = vec![
            RayTracingShaderGroupCreateInfo::General { general_shader: 0 },
            RayTracingShaderGroupCreateInfo::General { general_shader: 1 },
            RayTracingShaderGroupCreateInfo::TrianglesHit {
                closest_hit_shader: Some(2),
                any_hit_shader: None,
            },
        ];

        Self { stages, groups }
    }
}

impl fmt::Debug for closest_hit::ClosestHitPushConstants {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("closest_hit::ClosestHitPushConstants")
            .field("imageTextureCount", &self.imageTextureCount)
            .field("constantColourCount", &self.constantColourCount)
            .field("lambertianMaterialCount", &self.lambertianMaterialCount)
            .field("metalMaterialCount", &self.metalMaterialCount)
            .field("dielectricMaterialCount", &self.dielectricMaterialCount)
            .finish()
    }
}

impl fmt::Debug for ray_gen::RayGenPushConstants {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("closest_hit::RayGenPushConstants")
            .field("resolution", &self.resolution)
            .field("samplesPerPixel", &self.samplesPerPixel)
            .field("maxRayDepth", &self.maxRayDepth)
            .finish()
    }
}
