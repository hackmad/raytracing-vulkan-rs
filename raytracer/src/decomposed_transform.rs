use glam::{Mat4, Quat, Vec3};

/// Stores decomposed translation, rotation and scaling transformations.
///
/// Translation and scale are just vectors for containing information for each dimension. Rotation uses a unit
/// quaternion. These can then be interpolated separately and combined to give transformations for moving objects.
#[derive(Debug, Clone, Copy)]
pub struct DecomposedTransform {
    pub translation: Vec3,
    pub rotation: Quat, // unit quaternion
    pub scale: Vec3,
}

impl DecomposedTransform {
    /// Interpolate transformation at time t in [0, 1] between [self] starting at t = 0, and another
    /// [DecomposedTransform] ending transform at t = 1.
    pub fn lerp(&self, other: &DecomposedTransform, t: f32) -> Self {
        Self {
            translation: self.translation.lerp(other.translation, t),
            rotation: self.rotation.slerp(other.rotation, t),
            scale: self.scale.lerp(other.scale, t),
        }
    }

    /// Combine the individual transformations to give a 4x4 matrix.
    pub fn to_mat4(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }

    /// Combine the individual transformations and convert to a 3x4 matrix for Vulkan acceleration structures.
    pub fn to_vulkan_acc_mat(&self) -> [[f32; 4]; 3] {
        let m = self.to_mat4().transpose().to_cols_array_2d();
        [m[0], m[1], m[2]]
    }
}

impl From<&scene_file::Transform> for DecomposedTransform {
    /// Decompose a [scene_file::Transform].
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

impl From<Mat4> for DecomposedTransform {
    /// Decompose a [Mat4].
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
