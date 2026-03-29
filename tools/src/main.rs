use anyhow::Result;
use clap::{Parser, Subcommand};
use glam::{Mat4, Vec3};
use random::Random;
use scene_file::{
    Camera, Instance, InstanceType, Material, Primitive, Render, SceneFile, Sky, Texture,
    Transform, TransformType,
};

#[derive(Debug, Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Debug, Subcommand)]
enum Commands {
    GenFinalOneWeekend,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    Random::seed(485_674_845_675_491);

    match &cli.command {
        Some(Commands::GenFinalOneWeekend) => {
            generate_final_one_weekend_scene("assets/final-one-weekend.json", false)?;
            generate_final_one_weekend_scene("assets/final-one-weekend-motion-blur.json", true)?;
            generate_final_next_week_scene("assets/final-next-week.json")?;
        }
        None => {
            println!("Please specify a command");
        }
    }

    Ok(())
}

fn make_sphere_touch_ground(
    sphere_center: &[f32; 3],
    sphere_radius: f32,
    ground_sphere_center: &[f32; 3],
    ground_sphere_radius: f32,
) -> [f32; 3] {
    let g_center = Vec3::from_slice(ground_sphere_center);
    let dir = Vec3::from_slice(sphere_center) - g_center;
    const FUDGE: f32 = 0.035; // Pushes the sphere into the ground a little.
    (dir.normalize() * (ground_sphere_radius + sphere_radius - FUDGE) + g_center).to_array()
}

fn generate_final_one_weekend_scene(file_path: &str, do_motion_blur: bool) -> Result<()> {
    println!(
        "Generating Raytracing in One Weekend final scene file {file_path} {}",
        if do_motion_blur {
            "with motion blur"
        } else {
            "without motion blur"
        }
    );

    let mut primitives = vec![];
    let mut instances = vec![];
    let mut textures = vec![];
    let mut materials = vec![];
    let mut cameras = vec![];

    let green_texture = Texture::Constant {
        name: "green".to_string(),
        rgb: [0.2, 0.3, 0.1],
    };
    let white_texture = Texture::Constant {
        name: "pale-white".to_string(),
        rgb: [0.9, 0.9, 0.9],
    };
    let green_and_white_checker_texture = Texture::Checker {
        name: "green-and-white-checker".to_string(),
        scale: 0.32,
        even: green_texture.get_name().to_string(),
        odd: white_texture.get_name().to_string(),
    };

    let ground_material = Material::Lambertian {
        name: "ground".to_string(),
        albedo: green_and_white_checker_texture.get_name().to_string(),
    };

    let ground_center = [0.0, 1000.0, 0.0];
    let ground_radius = 1000.0;

    primitives.push(Primitive::UvSphere {
        name: "ground_sphere".to_string(),
        center: ground_center,
        radius: ground_radius,
        rings: 128,
        segments: 256,
        material: ground_material.get_name().to_string(),
    });
    textures.push(green_texture);
    textures.push(white_texture);
    textures.push(green_and_white_checker_texture);
    materials.push(ground_material);
    instances.push(Instance {
        name: "ground_sphere".to_string(),
        instance_type: InstanceType::Surface,
        transform: None,
    });

    let center_sphere_1 = Vec3::new(0.0, -1.0, 0.0);
    let center_sphere_2 = Vec3::from_array(make_sphere_touch_ground(
        &[-4.0, -1.0, 0.0],
        1.0,
        &ground_center,
        ground_radius,
    ));
    let center_sphere_3 = Vec3::from_array(make_sphere_touch_ground(
        &[4.0, -1.0, 0.0],
        1.0,
        &ground_center,
        ground_radius,
    ));

    let center_spheres_radius = 1.0;

    for a in -11..11 {
        for b in -11..11 {
            let choose_mat: f32 = Random::sample();

            let radius = 0.2;
            let mut center: [f32; 3];

            loop {
                center = [
                    a as f32 + 0.9 * Random::sample::<f32>(),
                    -radius,
                    b as f32 + 0.9 * Random::sample::<f32>(),
                ];
                center = make_sphere_touch_ground(&center, radius, &ground_center, ground_radius);

                let p_center = Vec3::from_slice(&center);

                let total_radius = center_spheres_radius + radius;
                if (p_center - center_sphere_1).length() > total_radius
                    && (p_center - center_sphere_2).length() > total_radius
                    && (p_center - center_sphere_3).length() > total_radius
                {
                    break;
                }
            }

            let (tex, material, transform) = if choose_mat < 0.8 {
                // diffuse
                let name = format!("diffuse_{a}_{b}");
                let t_albedo = Texture::Constant {
                    name: format!("tex_albedo_{name}"),
                    rgb: (Random::vec3() * Random::vec3()).to_array(),
                };
                let mat = Material::Lambertian {
                    name: format!("mat_{name}"),
                    albedo: t_albedo.get_name().to_string(),
                };
                let transform = if do_motion_blur {
                    Some(TransformType::Animated(
                        Transform {
                            translate: Some([0.0, Random::sample_in_range(-0.5, 0.0), 0.0]),
                            rotate: None,
                            scale: None,
                        },
                        Transform {
                            translate: Some([0.0, 0.0, 0.0]),
                            rotate: None,
                            scale: None,
                        },
                    ))
                } else {
                    None
                };
                (vec![t_albedo], mat, transform)
            } else if choose_mat < 0.95 {
                // metal
                let name = format!("metal_{a}_{b}");
                let t_albedo = Texture::Constant {
                    name: format!("tex_albedo_{name}"),
                    rgb: Random::vec3_in_range(0.5, 1.0).to_array(),
                };
                let t_fuzz = Texture::Constant {
                    name: format!("tex_fuzz_{name}"),
                    rgb: Random::vec3_in_range(0.0, 0.5).to_array(),
                };
                let mat = Material::Metal {
                    name: format!("mat_metal_{a}_{b}"),
                    albedo: t_albedo.get_name().to_string(),
                    fuzz: t_fuzz.get_name().to_string(),
                };
                (vec![t_albedo, t_fuzz], mat, None)
            } else {
                // glass
                let mat = Material::Dielectric {
                    name: format!("mat_dielectric_{a}_{b}"),
                    refraction_index: 1.5,
                };
                (vec![], mat, None)
            };

            let name = format!("sphere_{a}_{b}").to_string();
            primitives.push(Primitive::UvSphere {
                name: name.clone(),
                center,
                radius,
                rings: 32,
                segments: 64,
                material: material.get_name().to_string(),
            });
            instances.push(Instance {
                name,
                transform,
                instance_type: InstanceType::Surface,
            });

            textures.extend_from_slice(&tex);
            materials.push(material);
        }
    }

    let material1 = Material::Dielectric {
        name: "material1".to_string(),
        refraction_index: 1.5,
    };
    primitives.push(Primitive::UvSphere {
        name: "sphere1".to_string(),
        center: center_sphere_1.to_array(),
        radius: center_spheres_radius,
        rings: 64,
        segments: 128,
        material: material1.get_name().to_string(),
    });
    materials.push(material1);
    instances.push(Instance {
        name: "sphere1".to_string(),
        transform: None,
        instance_type: InstanceType::Surface,
    });

    let texture2 = Texture::Constant {
        name: "texture2".to_string(),
        rgb: [0.4, 0.2, 0.1],
    };
    let material2 = Material::Lambertian {
        name: "material2".to_string(),
        albedo: texture2.get_name().to_string(),
    };
    primitives.push(Primitive::UvSphere {
        name: "sphere2".to_string(),
        center: center_sphere_2.to_array(),
        radius: center_spheres_radius,
        rings: 64,
        segments: 128,
        material: material2.get_name().to_string(),
    });
    textures.push(texture2);
    materials.push(material2);
    instances.push(Instance {
        name: "sphere2".to_string(),
        transform: None,
        instance_type: InstanceType::Surface,
    });

    let texture3 = Texture::Constant {
        name: "texture3".to_string(),
        rgb: [0.7, 0.6, 0.5],
    };
    let texture4 = Texture::Constant {
        name: "texture4".to_string(),
        rgb: [0.0, 0.0, 0.0],
    };
    let material3 = Material::Metal {
        name: "material3".to_string(),
        albedo: texture3.get_name().to_string(),
        fuzz: texture4.get_name().to_string(),
    };
    primitives.push(Primitive::UvSphere {
        name: "sphere3".to_string(),
        center: center_sphere_3.to_array(),
        radius: center_spheres_radius,
        rings: 64,
        segments: 128,
        material: material3.get_name().to_string(),
    });
    textures.push(texture3);
    textures.push(texture4);
    materials.push(material3);
    instances.push(Instance {
        name: "sphere3".to_string(),
        transform: None,
        instance_type: InstanceType::Surface,
    });

    cameras.push(Camera::Perspective {
        name: "default".to_string(),
        eye: [13.0, -2.0, 3.0],
        look_at: [0.0, 0.0, 0.0],
        up: [0.0, 1.0, 0.0],
        fov_y: 20.0,
        z_near: 0.01,
        z_far: 100.0,
        focal_length: 10.0,
        aperture_size: 0.2,
    });

    let render = Render {
        camera: cameras[0].get_name().to_string(),
        samples_per_pixel: 4,
        sample_batches: 25,
        max_ray_depth: 50,
        aspect_ratio: 16.0 / 9.0,
    };

    let sky = Sky::VerticalGradient {
        factor: 0.5,
        top: [0.5, 0.7, 1.0],
        bottom: [1.0, 1.0, 1.0],
    };

    let scene_file = SceneFile {
        cameras,
        instances,
        materials,
        primitives,
        textures,
        sky,
        render,
    };
    scene_file.save_json(file_path)
}

fn generate_final_next_week_scene(file_path: &str) -> Result<()> {
    println!("Generating Raytracing in The Next Week final scene file {file_path}");

    let mut primitives = vec![];
    let mut instances = vec![];
    let mut textures = vec![];
    let mut materials = vec![];
    let mut cameras = vec![];

    // Ground with random boxes.
    let ground_texture = Texture::Constant {
        name: "ground".to_string(),
        rgb: [0.48, 0.83, 0.53],
    };
    let ground_material = Material::Lambertian {
        name: "ground".to_string(),
        albedo: ground_texture.get_name().to_string(),
    };

    const BOXES_PER_SIDE: usize = 20;
    for i in 0..BOXES_PER_SIDE {
        for j in 0..BOXES_PER_SIDE {
            let w = 100.0;
            let x0 = -1000.0 + i as f32 * w;
            let z0 = -1000.0 + j as f32 * w;
            let y0 = 0.0;
            let x1 = x0 + w;
            let y1 = Random::sample_in_range(-101.0, -1.0);
            let z1 = z0 + w;

            let name = format!("ground_box{i}_{j}");
            let ground_box = Primitive::Box {
                name: name.to_string(),
                corners: [[x0, y0, z0], [x1, y1, z1]],
                material: ground_material.get_name().to_string(),
            };
            primitives.push(ground_box);
            instances.push(Instance {
                name: name.to_string(),
                instance_type: InstanceType::Surface,
                transform: None,
            });
        }
    }
    textures.push(ground_texture);
    materials.push(ground_material);

    // Diffuse overhead light quad.
    let diffuse_light_texture = Texture::Constant {
        name: "diffuse_light".to_string(),
        rgb: [7.0, 7.0, 7.0],
    };

    let diffuse_light_material = Material::DiffuseLight {
        name: "diffuse_light".to_string(),
        emit: diffuse_light_texture.get_name().to_string(),
    };

    primitives.push(Primitive::Quad {
        name: "diffuse_light".to_string(),
        points: [
            [423.0, -554.0, 412.0],
            [123.0, -554.0, 412.0],
            [123.0, -554.0, 147.0],
            [423.0, -554.0, 147.0],
        ],
        normal: [0.0, 1.0, 0.0],
        uv: [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
        material: diffuse_light_texture.get_name().to_string(),
    });
    textures.push(diffuse_light_texture);
    materials.push(diffuse_light_material);
    instances.push(Instance {
        name: "diffuse_light".to_string(),
        instance_type: InstanceType::Surface,
        transform: None,
    });

    // Motion blurred sphere.
    let mblur_sphere_texture = Texture::Constant {
        name: "mblur_sphere_texture".to_string(),
        rgb: [0.7, 0.3, 0.1],
    };

    let mblur_sphere_material = Material::Lambertian {
        name: "mblur_sphere_material".to_string(),
        albedo: mblur_sphere_texture.get_name().to_string(),
    };
    let mblur_sphere = Primitive::UvSphere {
        name: "mblur_sphere".to_string(),
        center: [400.0, -400.0, 200.0],
        radius: 50.0,
        rings: 32,
        segments: 64,
        material: mblur_sphere_material.get_name().to_string(),
    };
    textures.push(mblur_sphere_texture);
    materials.push(mblur_sphere_material);
    primitives.push(mblur_sphere);
    instances.push(Instance {
        name: "mblur_sphere".to_string(),
        instance_type: InstanceType::Surface,
        transform: Some(TransformType::Animated(
            Transform {
                translate: None,
                rotate: None,
                scale: None,
            },
            Transform {
                translate: Some([30.0, 0.0, 0.0]),
                rotate: None,
                scale: None,
            },
        )),
    });

    // Glass sphere.
    let glass_sphere_material = Material::Dielectric {
        name: "glass_sphere_material".to_string(),
        refraction_index: 1.5,
    };
    let glass_sphere = Primitive::UvSphere {
        name: "glass_sphere".to_string(),
        center: [260.0, -150.0, 45.0],
        radius: 50.0,
        rings: 32,
        segments: 64,
        material: glass_sphere_material.get_name().to_string(),
    };
    primitives.push(glass_sphere);
    instances.push(Instance {
        name: "glass_sphere".to_string(),
        instance_type: InstanceType::Surface,
        transform: None,
    });

    // Metal sphere.
    let metal_sphere_albedo_texture = Texture::Constant {
        name: "metal_sphere_albedo_texture".to_string(),
        rgb: [0.8, 0.8, 0.9],
    };
    let metal_sphere_fuzz_texture = Texture::Constant {
        name: "metal_sphere_fuzz_texture".to_string(),
        rgb: [1.0, 1.0, 1.0],
    };

    let metal_sphere_material = Material::Metal {
        name: "metal_sphere_material".to_string(),
        albedo: metal_sphere_albedo_texture.get_name().to_string(),
        fuzz: metal_sphere_fuzz_texture.get_name().to_string(),
    };
    let metal_sphere = Primitive::UvSphere {
        name: "metal_sphere".to_string(),
        center: [0.0, -150.0, 145.0],
        radius: 50.0,
        rings: 32,
        segments: 64,
        material: metal_sphere_material.get_name().to_string(),
    };
    textures.push(metal_sphere_albedo_texture);
    textures.push(metal_sphere_fuzz_texture);
    materials.push(metal_sphere_material);
    primitives.push(metal_sphere);
    instances.push(Instance {
        name: "metal_sphere".to_string(),
        instance_type: InstanceType::Surface,
        transform: None,
    });

    // Constant density medium with sphere boundary where the boundary primitive is also instanced as a glass sphere.
    let phase_function_texture_1 = Texture::Constant {
        name: "phase_function_texture_1".to_string(),
        rgb: [0.2, 0.4, 0.9],
    };
    let phase_function_isotropic_material_1 = Material::Isotropic {
        name: "phase_function_isotropic_material_1".to_string(),
        phase_function: phase_function_texture_1.get_name().to_string(),
    };
    let glass_sphere_boundary_1 = Primitive::UvSphere {
        name: "glass_sphere_boundary_1".to_string(),
        center: [360.0, -150.0, 145.0],
        radius: 70.0,
        rings: 32,
        segments: 64,
        material: glass_sphere_material.get_name().to_string(),
    };
    primitives.push(glass_sphere_boundary_1);
    instances.push(Instance {
        name: "glass_sphere_boundary_1".to_string(),
        instance_type: InstanceType::Surface,
        transform: None,
    });
    instances.push(Instance {
        name: "glass_sphere_boundary_1".to_string(),
        instance_type: InstanceType::ConstantMedium {
            density: 0.2,
            phase_function: phase_function_isotropic_material_1.get_name().to_string(),
        },
        transform: None,
    });
    textures.push(phase_function_texture_1);
    materials.push(phase_function_isotropic_material_1);

    // Constant density medium with sphere boundary where the boundary primitive is NOT instanced.
    let phase_function_texture_2 = Texture::Constant {
        name: "phase_function_texture_2".to_string(),
        rgb: [1.0, 1.0, 1.0],
    };
    let phase_function_isotropic_material_2 = Material::Isotropic {
        name: "phase_function_isotropic_material_2".to_string(),
        phase_function: phase_function_texture_2.get_name().to_string(),
    };
    let glass_sphere_boundary_2 = Primitive::UvSphere {
        name: "glass_sphere_boundary_2".to_string(),
        center: [0.0, 0.0, 0.0],
        radius: 5000.0,
        rings: 32,
        segments: 64,
        material: glass_sphere_material.get_name().to_string(),
    };
    primitives.push(glass_sphere_boundary_2);
    instances.push(Instance {
        name: "glass_sphere_boundary_2".to_string(),
        instance_type: InstanceType::ConstantMedium {
            density: 0.0001,
            phase_function: phase_function_isotropic_material_2.get_name().to_string(),
        },
        transform: None,
    });
    textures.push(phase_function_texture_2);
    materials.push(phase_function_isotropic_material_2);

    materials.push(glass_sphere_material);

    // Earth
    let earth_texture = Texture::Image {
        name: "earth_texture".to_string(),
        path: "world.topo.bathy.200412.3x5400x2700.jpg".to_string(),
    };

    let earth_material = Material::Lambertian {
        name: "earth_material".to_string(),
        albedo: earth_texture.get_name().to_string(),
    };
    let earth = Primitive::UvSphere {
        name: "earth".to_string(),
        center: [400.0, -200.0, 400.0],
        radius: 100.0,
        rings: 32,
        segments: 64,
        material: earth_material.get_name().to_string(),
    };
    textures.push(earth_texture);
    materials.push(earth_material);
    primitives.push(earth);
    instances.push(Instance {
        name: "earth".to_string(),
        instance_type: InstanceType::Surface,
        transform: None,
    });

    // Perlin sphere
    let perlin_sphere_texture = Texture::Noise {
        name: "perlin_sphere_texture".to_string(),
        scale: 0.2,
    };

    let perlin_sphere_material = Material::Lambertian {
        name: "perlin_sphere_material".to_string(),
        albedo: perlin_sphere_texture.get_name().to_string(),
    };
    let perlin_sphere = Primitive::UvSphere {
        name: "perlin_sphere".to_string(),
        center: [220.0, -280.0, 300.0],
        radius: 80.0,
        rings: 32,
        segments: 64,
        material: perlin_sphere_material.get_name().to_string(),
    };
    textures.push(perlin_sphere_texture);
    materials.push(perlin_sphere_material);
    primitives.push(perlin_sphere);
    instances.push(Instance {
        name: "perlin_sphere".to_string(),
        instance_type: InstanceType::Surface,
        transform: None,
    });

    // Random spheres in box
    // Slightly different approach. Raytracing The Next Week does separate BVH but we have single
    // acceleration structure  So we can do one primitive + multiple instances.
    let white_spheres_texture = Texture::Constant {
        name: "white_spheres_texture".to_string(),
        rgb: [0.73, 0.73, 0.73],
    };
    let white_spheres_material = Material::Lambertian {
        name: "white_spheres_material".to_string(),
        albedo: white_spheres_texture.get_name().to_string(),
    };

    let white_sphere = Primitive::UvSphere {
        name: "white_sphere".to_string(),
        center: [0.0, 0.0, 0.0],
        radius: 10.0,
        rings: 32,
        segments: 64,
        material: white_spheres_material.get_name().to_string(),
    };

    // We will need to manually rotate the group of sphere locations ourselves about the y-axis
    // because we can't group our instances into a single hittable group.
    let rotation_matrix = Mat4::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 15_f32.to_radians());
    let translation = Vec3::new(-100.0, -270.0, 395.0);

    const NS: usize = 1000;
    for _j in 0..NS {
        let mut center = Random::vec3_in_range(0.0, 165.0);
        center.y *= -1.0; // Flipped y-coordinate.
        center = (rotation_matrix * center.extend(1.0)).truncate();
        center += translation;

        instances.push(Instance {
            name: white_sphere.get_name().to_string(),
            instance_type: InstanceType::Surface,
            transform: Some(TransformType::Static(Transform {
                translate: Some(center.into()),
                rotate: None,
                scale: None,
            })),
        });
    }
    primitives.push(white_sphere);
    textures.push(white_spheres_texture);
    materials.push(white_spheres_material);

    // Camera + Scene + Render settings
    cameras.push(Camera::Perspective {
        name: "default".to_string(),
        eye: [478.0, -278.0, -600.0],
        look_at: [278.0, -278.0, 0.0],
        up: [0.0, 1.0, 0.0],
        fov_y: 40.0,
        z_near: 0.01,
        z_far: 100.0,
        focal_length: 1.0,
        aperture_size: 0.0,
    });

    let render = Render {
        camera: cameras[0].get_name().to_string(),
        samples_per_pixel: 64,
        sample_batches: 2,
        max_ray_depth: 4,
        aspect_ratio: 1.0,
    };

    let sky = Sky::Solid {
        rgb: [0.0, 0.0, 0.0],
    };

    let scene_file = SceneFile {
        cameras,
        instances,
        materials,
        primitives,
        textures,
        sky,
        render,
    };
    scene_file.save_json(file_path)
}
