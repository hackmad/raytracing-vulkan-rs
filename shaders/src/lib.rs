mod camera;
mod material;
mod mesh;
mod push_constants;
mod sky;

use std::{io::Cursor, path::Path, sync::Arc};

use anyhow::Result;
use ash::vk;
pub use camera::*;
use log::{debug, info};
pub use material::*;
pub use mesh::*;
pub use push_constants::*;
pub use sky::*;
use vulkan::VulkanContext;

pub struct ShaderModules {
    context: Arc<VulkanContext>,
    pub ray_gen: vk::ShaderModule,
    pub ray_miss: vk::ShaderModule,
    pub closest_hit: vk::ShaderModule,
}

impl ShaderModules {
    pub fn load(context: Arc<VulkanContext>) -> Result<Self> {
        let ray_gen_code = read_shader_from_file(concat!(env!("OUT_DIR"), "/ray_gen.spv"));
        let ray_gen = create_shader_module(&context.device, &ray_gen_code)?;

        let ray_miss_code = read_shader_from_file(concat!(env!("OUT_DIR"), "/ray_miss.spv"));
        let ray_miss = create_shader_module(&context.device, &ray_miss_code)?;

        let closest_hit_code = read_shader_from_file(concat!(env!("OUT_DIR"), "/closest_hit.spv"));
        let closest_hit = create_shader_module(&context.device, &closest_hit_code)?;

        Ok(Self {
            context,
            ray_gen,
            ray_miss,
            closest_hit,
        })
    }
}

impl Drop for ShaderModules {
    fn drop(&mut self) {
        debug!("ShaderModules::drop()");
        unsafe {
            self.context.device.device_wait_idle().unwrap();

            self.context
                .device
                .destroy_shader_module(self.closest_hit, None);

            self.context
                .device
                .destroy_shader_module(self.ray_miss, None);

            self.context
                .device
                .destroy_shader_module(self.ray_gen, None);
        }
    }
}

fn create_shader_module(device: &ash::Device, code: &[u32]) -> Result<vk::ShaderModule> {
    let create_info = vk::ShaderModuleCreateInfo::default().code(code);
    let shader_module = unsafe { device.create_shader_module(&create_info, None)? };
    Ok(shader_module)
}

fn read_shader_from_file<P: AsRef<Path>>(path: P) -> Vec<u32> {
    info!("Loading shader file {}", path.as_ref().to_str().unwrap());
    let mut cursor = load(path);
    ash::util::read_spv(&mut cursor).unwrap()
}

fn load<P: AsRef<Path>>(path: P) -> Cursor<Vec<u8>> {
    use std::fs::File;
    use std::io::Read;

    let mut buf = Vec::new();
    let fullpath = Path::new("assets").join(path);
    let mut file = File::open(fullpath).unwrap();
    file.read_to_end(&mut buf).unwrap();
    Cursor::new(buf)
}
