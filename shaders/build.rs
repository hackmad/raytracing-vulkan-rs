use std::{env, fs, path::PathBuf};

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let shader_dir = PathBuf::from("glsl");

    println!("Shader directory: {shader_dir:?}");

    let compiler = shaderc::Compiler::new().unwrap();

    let mut options = shaderc::CompileOptions::new().unwrap();
    options.set_target_env(shaderc::TargetEnv::Vulkan, 1_4); // Vulkan 1.4
    options.set_target_spirv(shaderc::SpirvVersion::V1_6); // SPIR-V 1.6
    options.set_include_callback(|requested_source, _include_type, _source_name, _depth| {
        let include_path = shader_dir.join(requested_source);
        let content = fs::read_to_string(&include_path)
            .map_err(|e| format!("Failed to include file {}: {}", include_path.display(), e))?;
        Ok(shaderc::ResolvedInclude {
            resolved_name: include_path.display().to_string(),
            content,
        })
    });
    options.add_macro_definition("GL_EXT_ray_tracing", Some("1"));

    for entry in fs::read_dir(&shader_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();

        if let Some(ext) = path.file_name().and_then(|e| e.to_str()) {
            let shader_kind = match ext {
                "ray_gen.glsl" => shaderc::ShaderKind::RayGeneration,
                "closest_hit.glsl" => shaderc::ShaderKind::ClosestHit,
                "ray_miss.glsl" => shaderc::ShaderKind::Miss,
                _ => continue,
            };

            println!("Compiling {path:?}");

            let source = fs::read_to_string(&path).unwrap();
            let compiled_result = compiler
                .compile_into_spirv(
                    &source,
                    shader_kind,
                    path.file_name().unwrap().to_str().unwrap(),
                    "main",
                    Some(&options),
                )
                .unwrap();

            let output_path = out_dir
                .join(path.file_name().unwrap())
                .with_extension("spv");
            fs::write(output_path, compiled_result.as_binary_u8()).unwrap();
        }
    }

    println!("cargo:rerun-if-changed=shaders");
}
