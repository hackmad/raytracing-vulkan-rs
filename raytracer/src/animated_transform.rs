use glam::{Mat4, Quat, Vec3};

#[derive(Debug, Clone, Copy)]
pub struct AnimatedTransform {
    pub translation: Vec3,
    pub rotation: Quat, // unit quaternion
    pub scale: Vec3,
}

impl AnimatedTransform {
    pub fn lerp(&self, other: &AnimatedTransform, t: f32) -> Self {
        Self {
            translation: self.translation.lerp(other.translation, t),
            rotation: self.rotation.slerp(other.rotation, t),
            scale: self.scale.lerp(other.scale, t),
        }
    }

    pub fn to_mat4(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }

    /// Convert to a matrix for Vulkan TLAS
    pub fn to_vulkan_acc_mat(&self) -> [[f32; 4]; 3] {
        let m = self.to_mat4().transpose().to_cols_array_2d();
        [m[0], m[1], m[2]]
    }
}

impl From<&scene_file::Transform> for AnimatedTransform {
    fn from(value: &scene_file::Transform) -> Self {
        let translation = match value.translate {
            Some(v) => Vec3::from(v),
            None => Vec3::ZERO,
        };

        let scale = match value.scale {
            Some(v) => Vec3::from(v),
            None => Vec3::ONE,
        };

        let rotation = match value.rotate {
            Some(ref r) => {
                let axis = Vec3::from(r.axis).normalize_or_zero();
                let radians = r.degrees.to_radians();
                Quat::from_axis_angle(axis, radians)
            }
            None => Quat::IDENTITY,
        };

        Self {
            translation,
            rotation,
            scale,
        }
    }
}

impl From<Mat4> for AnimatedTransform {
    fn from(value: Mat4) -> Self {
        // Extract translation
        let translation = value.w_axis.truncate(); // last column (x,y,z)

        // Extract scale
        let scale = glam::Vec3::new(
            value.x_axis.truncate().length(),
            value.y_axis.truncate().length(),
            value.z_axis.truncate().length(),
        );

        // Remove scale to get rotation matrix
        let rotation_matrix = glam::Mat3::from_cols(
            value.x_axis.truncate() / scale.x,
            value.y_axis.truncate() / scale.y,
            value.z_axis.truncate() / scale.z,
        );

        // Convert rotation matrix to quaternion
        let rotation = glam::Quat::from_mat3(&rotation_matrix);

        Self {
            translation,
            rotation,
            scale,
        }
    }
}
