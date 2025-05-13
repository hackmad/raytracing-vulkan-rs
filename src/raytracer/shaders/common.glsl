#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference2 : enable

// Materials.
const uint MAT_PROP_TYPE_DIFFUSE = 0;

const uint MAT_PROP_VALUE_TYPE_NONE = 0;
const uint MAT_PROP_VALUE_TYPE_RGB = 1;
const uint MAT_PROP_VALUE_TYPE_TEXTURE = 2;

struct Material {
    uint propType;
    uint propValueType;
    int index;
};

// Vertex data.
struct MeshVertex {
    vec3 position;
    vec3 normal;
    vec2 texCoord;
};

// Mesh stores the pointers to the storage buffers.
layout(buffer_reference, scalar) buffer MeshVertcesRef {
    MeshVertex values[];
};
layout(buffer_reference, scalar) buffer MeshIndicesRef {
    uint values[];
};
layout(buffer_reference, scalar) buffer MeshMaterialsRef {
    Material values[];
};
struct Mesh {
    MeshVertcesRef verticesRef;
    MeshIndicesRef indicesRef;
    MeshMaterialsRef materialsRef;
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
