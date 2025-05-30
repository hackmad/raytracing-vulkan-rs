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
// Ray payload

struct RayPayload {
    uint rngState;
    bool isMissed;
    bool isScattered;
    vec3 scatteredRayOrigin;
    vec3 scatteredRayDirection;
    vec3 attenuation;
};

RayPayload initRayPayload(uint rngState) {
    RayPayload rp;
    rp.rngState = rngState;
    rp.isMissed = false;
    rp.isScattered = false;
    rp.scatteredRayOrigin = vec3(0.0);
    rp.scatteredRayDirection = vec3(0.0);
    rp.attenuation = vec3(0.0);
    return rp;
}


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
// General vector utility functions.

// Square of a vector's length.
float lengthSquared(vec2 v) {
    return v.x * v.x + v.y * v.y;
}
float lengthSquared(vec3 v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

// Return true if the vector is close to zero in all dimensions.
bool nearZero(vec3 v) {
    float s = 1e-8;
    return (abs(v.x) < s) && (abs(v.y) < s) && (abs(v.z) < s);
}

// --------------------------------------------------------------------------------
// Random number generator

uint initRNG(uvec2 pixel, uvec2 resolution) {
    return resolution.x * pixel.y + pixel.x;
}

// pcg32i_random_t with inc = 1.
uint stepRNG(uint rngState) {
    return rngState * 747796405 + 1;
}
float stepAndOutputRNGFloat(inout uint rngState) {
    // Steps the RNG and returns a floating-point value between 0 and 1 inclusive.
    // Condensed version of pcg_output_rxs_m_xs_32_32, with simple conversion to floating-point [0, 1].
    rngState  = stepRNG(rngState);
    uint word = ((rngState >> ((rngState >> 28) + 4)) ^ rngState) * 277803737;
    word      = (word >> 22) ^ word;
    return float(word) / 4294967295.0;
}

// Returns a random real in [0, 1).
float randomFloat(inout uint rngState) {
    return stepAndOutputRNGFloat(rngState);
}
vec2 randomVec2(inout uint rngState) {
    float v0 = randomFloat(rngState);
    float v1 = randomFloat(rngState);
    return vec2(v0, v1);
}
vec3 randomVec3(inout uint rngState) {
    float v0 = randomFloat(rngState);
    float v1 = randomFloat(rngState);
    float v2 = randomFloat(rngState);
    return vec3(v0, v1, v2);
}

// Returns a random real in [min, max).
float randomFloat(inout uint rngState, float min, float max) {
    return min + (max - min) * randomFloat(rngState);
}
vec2 randomVec2(inout uint rngState, float min, float max) {
    float v0 = randomFloat(rngState, min, max);
    float v1 = randomFloat(rngState, min, max);
    return vec2(v0, v1);
}
vec3 randomVec3(inout uint rngState, float min, float max) {
    float v0 = randomFloat(rngState, min, max);
    float v1 = randomFloat(rngState, min, max);
    float v2 = randomFloat(rngState, min, max);
    return vec3(v0, v1, v2);
}

vec3 randomUnitVec3(inout uint rngState) {
    while (true) {
        vec3 p = randomVec3(rngState, -1.0, 1.0);
        float lensq = lengthSquared(p);
        if (1e-160 < lensq && lensq <= 1) {
            return p / sqrt(lensq);
        }
    }
}

vec3 randomVec3OnHemisphere(inout uint rngState, vec3 normal) {
    vec3 onUnitSphere = randomUnitVec3(rngState);

    if (dot(onUnitSphere, normal) > 0.0) {
        // In the same hemisphere as the normal.
        return onUnitSphere;
    }

    return -onUnitSphere;
}

// Returns the vector to a random point in the [-.5, -.5] - [+.5, +.5] unit square.
vec2 sampleSquare(inout uint rngState) {
    return randomVec2(rngState) - vec2(0.5);
}


// --------------------------------------------------------------------------------
// Colour space conversions.

// Converts a colour from linear light gamma to sRGB gamma.
vec3 linearTosRGB(vec3 linearRGB)
{
    bvec3 cutoff = lessThan(linearRGB.rgb, vec3(0.0031308));
    vec3 higher = vec3(1.055) * pow(linearRGB.rgb, vec3(1.0 / 2.4)) - vec3(0.055);
    vec3 lower = linearRGB.rgb * vec3(12.92);
    return mix(higher, lower, cutoff);
}
vec4 linearTosRGB(vec4 linearRGB)
{
    vec3 colour = linearTosRGB(linearRGB.rgb);
    return vec4(colour, linearRGB.a);
}

// Converts a colour from sRGB gamma to linear light gamma.
vec3 sRGBToLinear(vec3 sRGB)
{
    bvec3 cutoff = lessThan(sRGB.rgb, vec3(0.04045));
    vec3 higher = pow((sRGB.rgb + vec3(0.055)) / vec3(1.055), vec3(2.4));
    vec3 lower = sRGB.rgb / vec3(12.92);
    return mix(higher, lower, cutoff);
}
vec4 sRGBToLinear(vec4 sRGB)
{
    vec3 colour = sRGBToLinear(sRGB.rgb);
    return vec4(colour, sRGB.a);
}

