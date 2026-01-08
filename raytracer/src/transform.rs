use crate::DecomposedTransform;

/// Stores decomposed transformations for static or moving mesh instances.
#[derive(Debug)]
pub enum Transform {
    /// Single transform for static mesh instances.
    Static(DecomposedTransform),

    /// Start and end transforms for moving mesh instances.
    Animated {
        start: DecomposedTransform,
        end: DecomposedTransform,
    },
}

impl From<scene_file::Matrix> for Transform {
    /// Decompose [scene_file::Matrix] to an equivalent [Transform].
    fn from(value: scene_file::Matrix) -> Self {
        match value {
            scene_file::Matrix::Static(mat) => Transform::Static(DecomposedTransform::from(mat)),

            scene_file::Matrix::Animated(mat1, mat2) => Transform::Animated {
                start: DecomposedTransform::from(mat1),
                end: DecomposedTransform::from(mat2),
            },
        }
    }
}
