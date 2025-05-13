use std::f32::consts::FRAC_PI_2;

use glam::{Mat4, Vec3};

/// Camera interface.
pub trait Camera {
    /// Update the rendered image size.
    fn update_image_size(&mut self, image_width: u32, image_height: u32);

    /// Returns the view matrix.
    fn get_view_matrix(&self) -> Mat4;

    /// Returns the inverse view matrix.
    fn get_view_inverse_matrix(&self) -> Mat4;

    /// Returns the projection matrix.
    fn get_projection_matrix(&self) -> Mat4;

    /// Returns the inverse projection matrix.
    fn get_projection_inverse_matrix(&self) -> Mat4;
}

/// Perspective camera.
pub struct PerspectiveCamera {
    eye: Vec3,
    look_at: Vec3,
    up: Vec3,
    z_near: f32,
    z_far: f32,
    proj: Mat4,
    view: Mat4,
}

impl PerspectiveCamera {
    /// Create a new perspective camera.
    pub fn new(
        eye: Vec3,
        look_at: Vec3,
        up: Vec3,
        z_near: f32,
        z_far: f32,
        image_width: u32,
        image_height: u32,
    ) -> Self {
        let aspect = image_width as f32 / image_height as f32;
        let proj = Mat4::perspective_rh(FRAC_PI_2, aspect, z_near, z_far);
        let view = Mat4::look_at_rh(eye, look_at, up);
        Self {
            eye,
            look_at,
            up,
            z_near,
            z_far,
            proj,
            view,
        }
    }
}

impl Camera for PerspectiveCamera {
    fn update_image_size(&mut self, image_width: u32, image_height: u32) {
        let aspect = image_width as f32 / image_height as f32;
        self.proj = Mat4::perspective_rh(FRAC_PI_2, aspect, self.z_near, self.z_far);
        self.view = Mat4::look_at_rh(self.eye, self.look_at, self.up);
    }

    fn get_view_matrix(&self) -> Mat4 {
        self.view
    }

    fn get_view_inverse_matrix(&self) -> Mat4 {
        self.view.inverse()
    }

    fn get_projection_matrix(&self) -> Mat4 {
        self.proj
    }

    fn get_projection_inverse_matrix(&self) -> Mat4 {
        self.proj.inverse()
    }
}
