#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_scalar_block_layout: enable

// --------------------------------------------------------------------------------
// Constants

const float PI = 3.14159265359;

// --------------------------------------------------------------------------------
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

// --------------------------------------------------------------------------------
// Vertex data.

struct MeshVertex {
    vec3 position;
    vec3 normal;
    vec2 texCoord;
};

// --------------------------------------------------------------------------------
// Mesh

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


// --------------------------------------------------------------------------------
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


// --------------------------------------------------------------------------------
// Random number generator.

// Tausworthe Generator.
// S1, S2, S3, and M are all constants, and z is part of the
// private per-thread generator state.
uint tausStep(uint z, int S1, int S2, int S3, uint M) {
    uint b = (((z << S1) ^ z) >> S2);
    return (((z & M) << S3) ^ b);    
}

// Linear Congruential Generator.
// A and C are constants
uint lcgStep(uint z, uint A, uint C) {
    return (A * z + C);    
}

// Hybrid Tausworthe + LCG Generator.
float randomFloat(inout uvec4 state) {
    state.x = tausStep(state.x, 13, 19, 12, 4294967294);
    state.y = tausStep(state.y, 2, 25, 4, 4294967288);
    state.z = tausStep(state.z, 3, 11, 17, 4294967280);
    state.w = lcgStep(state.w, 1664525, 1013904223);
    return 2.3283064365387e-10 * (state.x ^ state.y ^ state.z ^ state.w);
}
vec2 randomVec2(inout uvec4 state) {
    float v0 = randomFloat(state);
    float v1 = randomFloat(state);
    return vec2(v0, v1);
}
vec3 randomVec3(inout uvec4 state) {
    float v0 = randomFloat(state);
    float v1 = randomFloat(state);
    float v2 = randomFloat(state);
    return vec3(v0, v1, v2);
}

// Returns a random in the half open interval [min, max).
float randomFloat(inout uvec4 state, float min, float max) {
    float v = randomFloat(state);
    return min + (max - min) * v;
}
vec2 randomVec2(inout uvec4 state, float min, float max) {
    vec2 v = randomVec2(state);
    return min + (max - min) * v;
}
vec3 randomVec3(inout uvec4 state, float min, float max) {
    vec3 v = randomVec3(state);
    return min + (max - min) * v;
}

vec2 randomUnitVec2(inout uvec4 state) {
    while (true) {
        vec2 p = randomVec2(state, -1.0, 1.0);
        float lensq = length(p);

        if (lensq > 0.0 && lensq <= 1.0) {
            return p / sqrt(lensq);
        }
    }
}
vec3 randomUnitVec3(inout uvec4 state) {
    while (true) {
        vec3 p = randomVec3(state, -1.0, 1.0);
        float lensq = length(p);

        if (lensq > 0.0 && lensq <= 1.0) {
            return p / sqrt(lensq);
        }
    }
}

vec3 randomOnHemisphere(inout uvec4 state, vec3 normal) {
    vec3 onUnitSphere = randomUnitVec3(state);

    if (dot(onUnitSphere, normal) < 0.0) { 
        // Not in the same hemisphere as the normal.
        return -onUnitSphere;
    }

    return onUnitSphere;
}

// Box-Muller Transform
vec2 gaussianTransformVec2(inout uvec4 state) {
  float u0 = randomFloat(state);
  float u1 = randomFloat(state);

  float r = sqrt(-2.0 * log(u0));
  float theta = 2.0 * PI * u1;

  return vec2(r * sin(theta), r * cos(theta));
}


// --------------------------------------------------------------------------------
// Color space conversions.

// Converts a color from linear light gamma to sRGB gamma.
vec3 linearTosRGB(vec3 linearRGB)
{
    bvec3 cutoff = lessThan(linearRGB.rgb, vec3(0.0031308));
    vec3 higher = vec3(1.055) * pow(linearRGB.rgb, vec3(1.0 / 2.4)) - vec3(0.055);
    vec3 lower = linearRGB.rgb * vec3(12.92);
    return mix(higher, lower, cutoff);
}
vec4 linearTosRGB(vec4 linearRGB)
{
    vec3 color = linearTosRGB(linearRGB.rgb);
    return vec4(color, linearRGB.a);
}

// Converts a color from sRGB gamma to linear light gamma.
vec3 sRGBToLinear(vec3 sRGB)
{
    bvec3 cutoff = lessThan(sRGB.rgb, vec3(0.04045));
    vec3 higher = pow((sRGB.rgb + vec3(0.055)) / vec3(1.055), vec3(2.4));
    vec3 lower = sRGB.rgb / vec3(12.92);
    return mix(higher, lower, cutoff);
}
vec4 sRGBToLinear(vec4 sRGB)
{
    vec3 color = sRGBToLinear(sRGB.rgb);
    return vec4(color, sRGB.a);
}

