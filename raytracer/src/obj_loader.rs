use anyhow::Result;
use log::debug;
use shaders::MeshVertex;

/// Load a Wavefront OBJ file.
pub fn load_obj(path: &str) -> Result<Vec<(Vec<MeshVertex>, Vec<u32>)>> {
    let (models, _materials) = tobj::load_obj(path, &tobj::GPU_LOAD_OPTIONS)?;

    let mut result = vec![];

    for model in models.iter() {
        let mut vertices = vec![];
        let mut indices = vec![];

        let mesh = &model.mesh;

        for index in mesh.indices.iter() {
            let pos_offset = (3 * index) as usize;
            let tex_coord_offset = (2 * index) as usize;

            #[rustfmt::skip]
            let vertex = MeshVertex::new(
                [ mesh.positions[pos_offset], mesh.positions[pos_offset + 1], mesh.positions[pos_offset + 2] ], // p
                [ mesh.normals[pos_offset], mesh.normals[pos_offset + 1], mesh.normals[pos_offset + 2] ], // n
                [ mesh.texcoords[tex_coord_offset], 1.0 - mesh.texcoords[tex_coord_offset + 1] ], // uv
            );

            let vertex_index = vertices.len() as u32;

            vertices.push(vertex);
            indices.push(vertex_index);
        }

        debug!(
            "Vertex count: {}, Indices count: {}",
            vertices.len(),
            indices.len()
        );

        /*
        debug!("-------------------------------------------------------------------------------");
        debug!("     Position                     Normal                       UV");
        debug!("-------------------------------------------------------------------------------");
        for (i, v) in vertices.iter().enumerate() {
            debug!(
                "{i: >3}  [{: >7.4}, {: >7.4}, {: >7.4}]  [{: >7.4}, {: >7.4}, {: >7.4}]  [{:.4}, {:.4}]",
                v.p[0], v.p[1], v.p[2], v.n[0], v.n[1], v.n[2], v.uv[0], v.uv[1],
            );
        }
        debug!("-------------------------------------------------------------------------------");
        debug!("Indices {indices:?}");
        debug!("-------------------------------------------------------------------------------");
        */

        result.push((vertices, indices));
    }

    Ok(result)
}
