use anyhow::Result;
use clap::{Parser, Subcommand};
use glam::Vec3;
use random::Random;
use raytracer::{CameraType, MaterialPropertyValue, MaterialType, ObjectType, Render, SceneFile};

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

    Random::seed(485_674_958_675_300);

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

    let mut objects = vec![];
    let mut materials = vec![];
    let mut cameras = vec![];

    let ground_material = MaterialType::Lambertian {
        name: "ground".to_string(),
        albedo: MaterialPropertyValue::Rgb([0.5, 0.5, 0.5]),
    };

    let ground_center = [0.0, 1000.0, 0.0];
    let ground_radius = 1000.0;

    objects.push(ObjectType::UvSphere {
        name: "ground_sphere".to_string(),
        center: ground_center,
        radius: ground_radius,
        rings: 128,
        segments: 256,
        material: ground_material.get_name().to_string(),
    });

    materials.push(ground_material);

    for a in -11..11 {
        for b in -11..11 {
            let choose_mat: f32 = Random::sample();

            let radius = 0.2;
            let mut center = [
                a as f32 + 0.9 * Random::sample::<f32>(),
                -radius,
                b as f32 + 0.9 * Random::sample::<f32>(),
            ];
            center = make_sphere_touch_ground(&center, radius, &ground_center, ground_radius);

            let material = if choose_mat < 0.8 {
                // diffuse
                MaterialType::Lambertian {
                    name: format!("diffuse_{a}_{b}"),
                    albedo: MaterialPropertyValue::Rgb(
                        (Random::vec3() * Random::vec3()).to_array(),
                    ),
                }
            } else if choose_mat < 0.95 {
                // metal
                let fuzz = Random::sample_in_range(0.0, 0.5);

                MaterialType::Metal {
                    name: format!("metal_{a}_{b}"),
                    albedo: MaterialPropertyValue::Rgb(Random::vec3_in_range(0.5, 1.0).to_array()),
                    fuzz: MaterialPropertyValue::Rgb([fuzz, fuzz, fuzz]),
                }
            } else {
                // glass
                MaterialType::Dielectric {
                    name: format!("dielectric_{a}_{b}"),
                    refraction_index: 1.5,
                }
            };

            objects.push(ObjectType::UvSphere {
                name: format!("sphere_{a}_{b}").to_string(),
                center,
                radius,
                rings: 32,
                segments: 64,
                material: material.get_name().to_string(),
            });

            materials.push(material);
        }
    }

    let material1 = MaterialType::Dielectric {
        name: "material1".to_string(),
        refraction_index: 1.5,
    };
    objects.push(ObjectType::UvSphere {
        name: "sphere1".to_string(),
        center: [0.0, -1.0, 0.0],
        radius: 1.0,
        rings: 64,
        segments: 128,
        material: material1.get_name().to_string(),
    });
    materials.push(material1);

    let mut center2 = [-4.0, -1.0, 0.0];
    center2 = make_sphere_touch_ground(&center2, 1.0, &ground_center, ground_radius);

    let material2 = MaterialType::Lambertian {
        name: "material2".to_string(),
        albedo: MaterialPropertyValue::Rgb([0.4, 0.2, 0.1]),
    };
    objects.push(ObjectType::UvSphere {
        name: "sphere2".to_string(),
        center: center2,
        radius: 1.0,
        rings: 64,
        segments: 128,
        material: material2.get_name().to_string(),
    });
    materials.push(material2);

    let mut center3 = [4.0, -1.0, 0.0];
    center3 = make_sphere_touch_ground(&center3, 1.0, &ground_center, ground_radius);

    let material3 = MaterialType::Metal {
        name: "material3".to_string(),
        albedo: MaterialPropertyValue::Rgb([0.7, 0.6, 0.5]),
        fuzz: MaterialPropertyValue::Rgb([0.0, 0.0, 0.0]),
    };
    objects.push(ObjectType::UvSphere {
        name: "sphere3".to_string(),
        center: center3,
        radius: 1.0,
        rings: 64,
        segments: 128,
        material: material3.get_name().to_string(),
    });
    materials.push(material3);

    cameras.push(CameraType::Perspective {
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
    };

    let scene_file = SceneFile {
        cameras,
        materials,
        objects,
        render,
    };
    scene_file.save_json("assets/final-one-weekend.json")
}
