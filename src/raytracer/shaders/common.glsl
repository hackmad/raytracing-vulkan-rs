#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable

struct Vertex {
    vec3 position;
    vec3 normal;
    vec2 texCoord;
};

layout(buffer_reference, scalar) buffer Vertices {
    Vertex values[];
};
layout(buffer_reference, scalar) buffer Indices {
    uint values[];
};
struct Mesh {
    Vertices vertices;
    Indices indices;
};

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
