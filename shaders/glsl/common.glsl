#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_scalar_block_layout: enable

// --------------------------------------------------------------------------------
// Constants

const float PI = 3.14159265359;
const float TWO_PI = 2.0 * PI;
const float PI_OVER_2 = PI / 2.0;
const float PI_OVER_4 = PI / 4.0;

// --------------------------------------------------------------------------------
// Materials.

const uint MAT_TYPE_NONE = 0;
const uint MAT_TYPE_LAMBERTIAN = 1;
const uint MAT_TYPE_METAL = 2;
const uint MAT_TYPE_DIELECTRIC = 3;
const uint MAT_TYPE_DIFFUSE_LIGHT = 4;

const uint MAT_PROP_VALUE_TYPE_RGB = 0;
const uint MAT_PROP_VALUE_TYPE_IMAGE = 1;
const uint MAT_PROP_VALUE_TYPE_CHECKER = 2;
const uint MAT_PROP_VALUE_TYPE_NOISE = 3;

struct MaterialPropertyValue {
    uint propValueType;
    uint index;
};

struct LambertianMaterial {
    MaterialPropertyValue albedo;
};

struct MetalMaterial {
    MaterialPropertyValue albedo;
    MaterialPropertyValue fuzz;
};

struct DielectricMaterial {
    float refractionIndex;
};

struct DiffuseLightMaterial {
    MaterialPropertyValue emit;
};

struct CheckerTexture {
    float scale;
    MaterialPropertyValue odd;
    MaterialPropertyValue even;
};

struct NoiseTexture {
    float scale;
};

// --------------------------------------------------------------------------------
// Sky.

const uint SKY_TYPE_NONE = 0;
const uint SKY_TYPE_SOLID = 1;
const uint SKY_TYPE_VERTICAL_GRADIENT = 2;

struct Sky {
    vec3 solid;     // Solid colour.

    uint skyType;   // Sky type.
    
    vec3 vTop;      // Vertical gradient top colour;
    float vFactor;  // Vertical gradient factor.
    vec3 vBottom;   // Vertical gradient bottom colour;
};

// --------------------------------------------------------------------------------
// Mesh

// NOTE: The order of fields below will ensure data is aligned/packed correctly and
// we can avoid having to use padding fields.
struct MeshVertex {
    vec3 p;  // position
    float u; // u- texture coordinate
    vec3 n;  // normal
    float v; // v- texture coordinate
};


struct Mesh {
    uint vertexBufferSize;
    uint indexBufferSize;
    uint materialType;
    uint materialIndex;
};

// --------------------------------------------------------------------------------
// Hit record

struct HitRecord {
    MeshVertex meshVertex;
    bool isFrontFace;
    vec3 normal; // Points against the incident ray.
};


// --------------------------------------------------------------------------------
// Ray payload

struct RayPayload {
    uint rngState;
    bool isMissed;
    bool isScattered;
    vec3 scatteredRayOrigin;
    vec3 scatteredRayDirection;
    vec3 scatterColour;
    vec3 emissionColour;
};

RayPayload initRayPayload(uint rngState) {
    RayPayload rp;
    rp.rngState = rngState;
    rp.isMissed = false;
    rp.isScattered = false;
    rp.scatteredRayOrigin = vec3(0.0);
    rp.scatteredRayDirection = vec3(0.0);
    rp.scatterColour = vec3(0.0);
    rp.emissionColour = vec3(0.0);
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

bool isFrontFace(vec3 rayDirection, vec3 outwardNormal) {
    return dot(rayDirection, outwardNormal) < 0.0;
}

// --------------------------------------------------------------------------------
// Random number generator

uint initRNG(uint sampleBatch, uvec2 pixel, uvec2 resolution) {
    return (sampleBatch * resolution.y + pixel.y) * resolution.x + pixel.x;
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

// Gaussian filter: https://nvpro-samples.github.io/vk_mini_path_tracer/extras.html#gaussianfilterantialiasing 
vec2 randomGaussian(inout uint rngState) {
    // Almost uniform in (0,1] - make sure the value is never 0:
    const float u1    = max(1e-38, stepAndOutputRNGFloat(rngState));
    const float u2    = stepAndOutputRNGFloat(rngState);  // In [0, 1]
    const float r     = sqrt(-2.0 * log(u1));
    const float theta = TWO_PI * u2;  // Random in [0, 2pi]
    return r * vec2(cos(theta), sin(theta));
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

// Uses rejection sampling.
vec2 randomVec2InUnitDisk(inout uint rngState) {
    while (true) {
        vec2 p = randomVec2(rngState, -1.0, 1.0);
        if (lengthSquared(p) < 1.0) {
            return p;
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

// Box filter. Returns the vector to a random point in the [-.5, -.5] - [+.5, +.5] unit square.
vec2 sampleSquare(inout uint rngState) {
    return randomVec2(rngState) - vec2(0.5);
}

vec2 sampleUniformDiskConcentric(inout uint rngState) {
    vec2 u = randomVec2(rngState);

    // Map u to and handle degeneracy at the origin.
    vec2 uOffset = 2.0 * u - vec2(1.0);
    if (uOffset.x == 0.0 && uOffset.y == 0.0) {
        return vec2(0.0);
    }

    // Apply concentric mapping to point.
    float theta;
    float r;
    if (abs(uOffset.x) > abs(uOffset.y)) {
        r = uOffset.x;
        theta = PI_OVER_4 * (uOffset.y / uOffset.x);
    } else {
        r = uOffset.y;
        theta = PI_OVER_2 - PI_OVER_4 * (uOffset.x / uOffset.y);
    }
    return r * vec2(cos(theta), sin(theta));
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

float linearToGamma(float v) {
    if (v > 0) {
        return sqrt(v);
    }
    return 0;
}
vec3 linearToGamma(vec3 linearRGB) {
    return vec3(
         clamp(linearToGamma(linearRGB.x), 0, 1),
         clamp(linearToGamma(linearRGB.y), 0, 1),
         clamp(linearToGamma(linearRGB.z), 0, 1)
    ); 
}
