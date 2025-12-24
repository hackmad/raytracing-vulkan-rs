#version 460

// Use one big triangle with 2 vertices outside the usual clip space range.
// The texture coordinates used in the fragment shader will line up for the
// entire viewport that fits perfectly inside this triangle.
vec2 positions[3] = vec2[](
    vec2(-1.0, -1.0), // bottom-left
    vec2( 3.0, -1.0), // bottom-right (far outside clip space)
    vec2(-1.0,  3.0)  // top-left (far outside clip space)
);

void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}

