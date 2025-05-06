#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference2 : enable

// Vertex data.
struct MeshVertex {
    vec3 position;
    vec3 normal;
    vec2 texCoord;
};


// Pointer to a storage buffer containing the mesh vertex data.
layout(buffer_reference, scalar) buffer MeshVertcesRef {
    MeshVertex values[];
};

// Pointer to a storage buffer containing the mesh indices.
layout(buffer_reference, scalar) buffer MeshIndicesRef {
    uint values[];
};

// Mesh stores the pointers to the storage buffers containing vertex data and indices.
struct Mesh {
    MeshVertcesRef verticesRef;
    MeshIndicesRef indicesRef;
};


// Map a value from [fromMin, fromMax] to [toMin, toMax].
float map(float value, float fromMin, float fromMax, float toMin, float toMax) {
  return toMin + (toMax - toMin) * (value - fromMin) / (fromMax - fromMin);
}
vec2 map(vec2 value, float fromMin, float fromMax, float toMin, float toMax) {
  return toMin + (toMax - toMin) * (value - fromMin) / (fromMax - fromMin);
}
vec3 map(vec3 value, float fromMin, float fromMax, float toMin, float toMax) {
  return toMin + (toMax - toMin) * (value - fromMin) / (fromMax - fromMin);
}
vec4 map(vec4 value, float fromMin, float fromMax, float toMin, float toMax) {
  return toMin + (toMax - toMin) * (value - fromMin) / (fromMax - fromMin);
}
