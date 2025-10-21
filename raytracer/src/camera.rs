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

    /// Returns the focal length of the lens.
    fn get_focal_length(&self) -> f32;

    /// Returns the aperture size of the lens.
    fn get_aperture_size(&self) -> f32;
}

/// Perspective camera.
pub struct PerspectiveCamera {
    eye: Vec3,
    look_at: Vec3,
    up: Vec3,
    fov_y: f32, // Vertical FOV in radians.
    z_near: f32,
    z_far: f32,
    proj: Mat4,
    view: Mat4,
    focal_length: f32,
    aperture_size: f32,
}

impl PerspectiveCamera {
    /// Create a new perspective camera.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        eye: Vec3,
        look_at: Vec3,
        up: Vec3,
        fov_y: f32,
        z_near: f32,
        z_far: f32,
        focal_length: f32,
        aperture_size: f32,
        image_width: u32,
        image_height: u32,
    ) -> Self {
        let aspect = image_width as f32 / image_height as f32;
        let proj = Mat4::perspective_rh(fov_y, aspect, z_near, z_far);
        let view = Mat4::look_at_rh(eye, look_at, up);
        Self {
            eye,
            look_at,
            up,
            fov_y,
            z_near,
            z_far,
            focal_length,
            aperture_size,
            proj,
            view,
        }
    }
}

impl Camera for PerspectiveCamera {
    fn update_image_size(&mut self, image_width: u32, image_height: u32) {
        let aspect = image_width as f32 / image_height as f32;
        self.proj = Mat4::perspective_rh(self.fov_y, aspect, self.z_near, self.z_far);
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

    fn get_focal_length(&self) -> f32 {
        self.focal_length
    }

    fn get_aperture_size(&self) -> f32 {
        self.aperture_size
    }
}

pub fn create_camera(
    scene_camera: &scene_file::Camera,
    image_width: u32,
    image_height: u32,
) -> Box<dyn crate::Camera> {
    match scene_camera {
        scene_file::Camera::Perspective {
            name: _,
            eye,
            look_at,
            up,
            fov_y,
            z_near,
            z_far,
            focal_length,
            aperture_size,
        } => Box::new(PerspectiveCamera::new(
            Vec3::from_slice(eye),
            Vec3::from_slice(look_at),
            Vec3::from_slice(up),
            fov_y.to_radians(),
            *z_near,
            *z_far,
            *focal_length,
            *aperture_size,
            image_width,
            image_height,
        )),
    }
}
