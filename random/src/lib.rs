//! # Random
//!
//! A library for generating random numbers.

#![allow(dead_code)]

use glam::Vec3;
use rand::distr::uniform::SampleUniform;
use rand::distr::{Distribution, StandardUniform};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::cell::RefCell;
use std::f32::consts::PI;

thread_local! {
    /// Create a new thread local seedable random number generator initialized
    /// with a random seed.
    static RNG: RefCell<ChaCha20Rng> = {
        let rng= ChaCha20Rng::from_os_rng();
        RefCell::new(rng)
    }
}

/// Wraps some common random sample generation routines using a thread_rng().
pub struct Random {}

impl Random {
    /// Set the seed for the random number generator.
    ///
    /// * `s` - The seed.
    pub fn seed(s: u64) {
        RNG.with(|rng| *rng.borrow_mut() = SeedableRng::seed_from_u64(s))
    }

    /// Returns a random value.
    pub fn sample<T>() -> T
    where
        StandardUniform: Distribution<T>,
    {
        RNG.with(|rng| rng.borrow_mut().random::<T>())
    }

    /// Returns `n` random floating point values in [0, 1].
    ///
    /// * `n` - Number of samples.
    pub fn samples<T>(n: usize) -> Vec<T>
    where
        StandardUniform: Distribution<T>,
    {
        RNG.with(|rng| {
            let mut r = rng.borrow_mut();
            (0..n).map(|_| r.random::<T>()).collect()
        })
    }

    /// Returns a random value in [`min`, `max`).
    ///
    /// * `min` - Minimum bound
    /// * `max` - Maximum bound
    pub fn sample_in_range<T>(min: T, max: T) -> T
    where
        T: SampleUniform + PartialOrd,
    {
        RNG.with(|rng| {
            let mut r = rng.borrow_mut();
            r.random_range(min..max)
        })
    }

    /// Returns `n` random values in [`min`, `max`).
    ///
    /// * `n` - Number of samples.
    /// * `min` - Minimum bound
    /// * `max` - Maximum bound
    pub fn samples_in_range<T>(n: usize, min: T, max: T) -> Vec<T>
    where
        T: SampleUniform + PartialOrd + Copy,
    {
        RNG.with(|rng| {
            let mut r = rng.borrow_mut();
            (0..n).map(|_| r.random_range(min..max)).collect()
        })
    }

    /// Return a random value via the [`StandardUniform`] distribution.
    pub fn random<T>() -> T
    where
        StandardUniform: Distribution<T>,
    {
        RNG.with(|rng| {
            let mut r = rng.borrow_mut();
            r.random::<T>()
        })
    }

    /// Returns a random vector with random components in [0, 1].
    pub fn vec3() -> Vec3 {
        RNG.with(|rng| {
            let mut r = rng.borrow_mut();
            Vec3::new(r.random::<f32>(), r.random::<f32>(), r.random::<f32>())
        })
    }

    /// Returns a random unit vector by picking points on the unit sphere
    /// and then normalizing it.
    pub fn vec3_in_range(min: f32, max: f32) -> Vec3 {
        RNG.with(|rng| {
            let mut r = rng.borrow_mut();
            Vec3::new(
                r.random_range(min..max),
                r.random_range(min..max),
                r.random_range(min..max),
            )
        })
    }

    /// Returns a random vector within the unit sphere. This vector is not
    /// normalized.
    pub fn vec3_in_unit_sphere() -> Vec3 {
        RNG.with(|rng| {
            let mut r = rng.borrow_mut();
            loop {
                let p = Vec3::new(
                    r.random_range(-1.0..1.0),
                    r.random_range(-1.0..1.0),
                    r.random_range(-1.0..1.0),
                );
                if p.length_squared() < 1.0 {
                    break p;
                }
            }
        })
    }

    /// Returns a random unit vector by picking points on the unit sphere
    /// and then normalizing it.
    pub fn unit_vec3() -> Vec3 {
        RNG.with(|rng| {
            let mut r = rng.borrow_mut();
            let a: f32 = r.random_range(0.0..(2.0 * PI));
            let z: f32 = r.random_range(-1.0..1.0);
            let r: f32 = (1.0 - z * z).sqrt();
            Vec3::new(r * a.cos(), r * a.sin(), z)
        })
    }

    /// Returns a random vector with uniform scatter direction for all angles
    /// away from a hit point, with no dependence on the angle from the normal.
    ///
    /// * `normal` - THe surface normal.
    pub fn vec3_in_hemisphere(normal: Vec3) -> Vec3 {
        let in_unit_sphere = Random::vec3_in_unit_sphere();
        if in_unit_sphere.dot(normal) > 0.0 {
            // In the same hemisphere as the normal
            in_unit_sphere
        } else {
            -in_unit_sphere
        }
    }

    /// Returns a random point inside unit disk in the xy-plane.
    pub fn vec3_in_unit_disk() -> Vec3 {
        RNG.with(|rng| {
            let mut r = rng.borrow_mut();
            loop {
                let p = Vec3::new(r.random_range(-1.0..1.0), r.random_range(-1.0..1.0), 0.0);
                if p.length_squared() < 1.0 {
                    break p;
                }
            }
        })
    }

    /// Shuffle a `Vec<T>` in place.
    ///
    /// * `v` - Vector to shuffle.
    pub fn permute<T>(v: &mut [T])
    where
        T: Copy,
    {
        RNG.with(|rng| {
            let mut r = rng.borrow_mut();
            for i in (1..v.len()).rev() {
                let target = r.random_range(0..i);

                let (x, y) = (v[i], v[target]);

                v[i] = y;
                v[target] = x;
            }
        })
    }

    /// Returns a random vector based on p(direction) = cos(θ) / π.
    pub fn cosine_direction() -> Vec3 {
        RNG.with(|rng| {
            let mut r = rng.borrow_mut();

            let r1 = r.random::<f32>();
            let r2 = r.random::<f32>();
            let z = (1.0 - r2).sqrt();

            let phi = 2.0 * PI * r1;

            let r2_sqrt = r2.sqrt();
            let x = phi.cos() * r2_sqrt;
            let y = phi.sin() * r2_sqrt;

            Vec3::new(x, y, z)
        })
    }

    // Return a random vector uniformly sampled from a sphere’s solid angle
    // from a point outside the sphere
    //
    // * `distance_squared` - Square of distance to a point from sphere center.
    pub fn vec3_to_sphere(radius: f32, distance_squared: f32) -> Vec3 {
        RNG.with(|rng| {
            let mut r = rng.borrow_mut();

            let r1 = r.random::<f32>();
            let r2 = r.random::<f32>();

            let r_squared_over_d_squared = radius * radius / distance_squared;
            let z = 1.0 + r2 * ((1.0 - r_squared_over_d_squared).sqrt() - 1.0);

            let phi = 2.0 * PI * r1;

            let sqrt_one_minus_z_squared = (1.0 - z * z).sqrt();
            let x = phi.cos() * sqrt_one_minus_z_squared;
            let y = phi.sin() * sqrt_one_minus_z_squared;

            Vec3::new(x, y, z)
        })
    }
}
