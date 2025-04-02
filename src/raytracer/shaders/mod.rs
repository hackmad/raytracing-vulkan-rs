use std::sync::Arc;

use vulkano::{device::Device, shader::EntryPoint};

pub mod ray_gen {
    vulkano_shaders::shader! {
        ty: "raygen",
        path: "src/raytracer/shaders/ray_gen.glsl",
        vulkan_version: "1.3",
    }
}

pub mod closest_hit {
    vulkano_shaders::shader! {
        ty: "closesthit",
        path: "src/raytracer/shaders/closest_hit.glsl",
        vulkan_version: "1.3",
    }
}

pub mod ray_miss {
    vulkano_shaders::shader! {
        ty: "miss",
        path: "src/raytracer/shaders/ray_miss.glsl",
        vulkan_version: "1.3",
    }
}

pub struct ShaderModules {
    pub ray_gen: EntryPoint,
    pub closest_hit: EntryPoint,
    pub ray_miss: EntryPoint,
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

        Self {
            ray_gen,
            closest_hit,
            ray_miss,
        }
    }
}
