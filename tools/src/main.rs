use anyhow::Result;
use clap::{Parser, Subcommand};
use glam::Vec3;
use random::Random;
use raytracer::{Camera, Material, Primitive, Render, SceneFile, Sky, Texture};

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
            generate_final_one_weekend_scene()?;
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

fn generate_final_one_weekend_scene() -> Result<()> {
    println!("Generating Raytracing in One Weekend final scene file");

    let mut primitives = vec![];
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

            let (tex, material) = if choose_mat < 0.8 {
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
                (vec![t_albedo], mat)
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
                (vec![t_albedo, t_fuzz], mat)
            } else {
                // glass
                let mat = Material::Dielectric {
                    name: format!("mat_dielectric_{a}_{b}"),
                    refraction_index: 1.5,
                };
                (vec![], mat)
            };

            primitives.push(Primitive::UvSphere {
                name: format!("sphere_{a}_{b}").to_string(),
                center,
                radius,
                rings: 32,
                segments: 64,
                material: material.get_name().to_string(),
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
        samples_per_pixel: 64,
        sample_batches: 2,
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
        textures,
        materials,
        primitives,
        sky,
        render,
    };
    scene_file.save_json("assets/final-one-weekend.json")
}
