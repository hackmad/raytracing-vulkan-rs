#version 460
#extension GL_EXT_ray_tracing : require

#include "common.glsl"
#include "perlin.glsl"

layout(location = 0) rayPayloadEXT RayPayload rp;
layout(location = 1) rayPayloadEXT bool isShadowed;

layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;

layout(set = 1, binding = 0) uniform Camera {
    mat4  viewProj;     // Camera view * projection
    mat4  viewInverse;  // Camera inverse view matrix
    mat4  projInverse;  // Camera inverse projection matrix
    float focalLength;  // Focal length of lens.
    float apertureSize; // Aperture size (diameter of lens).
} camera;

layout(set = 2, binding = 0, rgba32f) uniform image2D image;

layout(set = 8, binding = 0) uniform Sky {
    vec3 solid;     // Solid colour.

    uint skyType;   // Sky type.

    vec3 vTop;      // Vertical gradient top colour;
    float vFactor;  // Vertical gradient factor.
    vec3 vBottom;   // Vertical gradient bottom colour;
} sky;

// NOTES:
//
// Make sure to check layout offsets for push constants in each of the shader files.
// The order of these matter so make sure they are consistent across all shaders.
//
// See https://nvpro-samples.github.io/vk_mini_path_tracer/extras.html#moresamples.
// It explains not exceeding 64 samples per pixel and 32 batches to avoid timeouts and long renders.
// We now do progressive rendering so 64 samples per pixel is still good and you can do higher number
// of batches especially for motion blur.
//
// The batchRayTime is included here for correctness. However, it is used when building the acceleration
// structures with interpolated transformations for moving objects. At the moment, it is not used for
// anything but included for correctness. Later we could use it for time dependent features such as:
// - Animated materials
// - Time-varying emission
// - Procedural textures
// - Camera motion blur
// - Light sampling
// - BSDFs with time dependence
// - Random number decorrelation
layout(push_constant) uniform PushConstants {
    layout(offset =  0) uvec2 resolution;
    layout(offset =  8) uint  samplesPerPixel;
    layout(offset = 12) uint  sampleBatch;
    layout(offset = 16) uint  maxRayDepth;
    layout(offset = 20) float batchRayTime;
} pc;


vec3 getBackgroundColour(Ray ray) {
    vec3 unitDirection = normalize(ray.direction);
    float a = 0.5 * (unitDirection.y + 1.0);

    switch (sky.skyType) {
        case SKY_TYPE_SOLID:
            return sky.solid;
        case SKY_TYPE_VERTICAL_GRADIENT:
            return mix(sky.vTop, sky.vBottom, sky.vFactor);
            break;
        default:
            return vec3(0.0);
    }
}

vec3 rayColour(inout uint rngState, Ray ray, float tMin, float tMax, uint rayFlags) {
    vec3 accumulated = vec3(0.0);
    vec3 throughput  = vec3(1.0);

    for (uint depth = pc.maxRayDepth; depth > 0; --depth) {
        rp = initRayPayload(rngState, pc.batchRayTime);

        // sbtRecordOffset, sbtRecordStride control how the hitGroupId (VkAccelerationStructureInstanceKHR::
        // instanceShaderBindingTablerecordOffset) of each instance is used to look up a hit group in the 
        // SBT's hit group array. Since we only have one hit group, both are set to 0.
        //
        // missIndex is the index, within the miss shader group array of the SBT to call if no intersection is found.
        traceRayEXT(
                topLevelAS,    // acceleration structure
                rayFlags,      // rayFlags
                0xFF,          // cullMask
                0,             // sbtRecordOffset
                0,             // sbtRecordStride
                0,             // missIndex
                ray.origin,    // ray origin
                tMin,          // ray min range
                ray.direction, // ray direction
                tMax,          // ray max range
                0);            // payload (location = 0)

        rngState = rp.rngState;

        // Closest hit and miss shader will set rp.isMissed.
        if (rp.isMissed) {
            vec3 bgColour = getBackgroundColour(ray);
            accumulated += throughput * bgColour;
            break;
        }

        // Add emmission contribution.
        accumulated += throughput * rp.erec.emissionColour;

        // If ray is not scattered we are done with this path.
        if (!rp.srec.isScattered) {
            break;
        }

        // Return early if scattering PDF was not evaluated.
        if (rp.srec.skipPdf) {
            // Update throughput.
            throughput *= rp.srec.attenuation;

            // Use scattered ray computed from material scattering for next depth.
            ray = rp.srec.skipPdfRay;

            continue;
        }

        // Update throughput.
        throughput *= rp.srec.attenuation;

        // Calculate ray for next depth.
        ray = Ray(rp.rec.meshVertex.p, normalize(rp.srec.scatterDirection), ray.time);
    }

    return accumulated;
}

Ray getRay(inout uint rngState, vec2 pixelCenter, int si, int sj, float recipSqrtSpp) {
    const vec2 offset = sampleSquareStratified(rngState, si, sj, recipSqrtSpp);
    const vec2 offsetPixelCenter = pixelCenter + offset;

    const vec2 screenUV = offsetPixelCenter / vec2(gl_LaunchSizeEXT.xy);
    vec2 d = screenUV * 2.0 - 1.0;

    vec4 origin = camera.viewInverse * vec4(0.0, 0.0, 0.0, 1.0);
    vec4 target = camera.projInverse * vec4(d.x, d.y, 1.0, 1.0);
    vec4 direction = camera.viewInverse * vec4(normalize(target.xyz), 0.0);

    if (camera.apertureSize > 0.0) {
        vec4 focalPoint = vec4(camera.focalLength * normalize(target.xyz), 1.0);

        vec2 randomLensPos = sampleUniformDiskConcentric(rngState) * camera.apertureSize / 2.0;
        origin.xy += vec2(randomLensPos.x * d.x, randomLensPos.y * d.y);

        direction = vec4((normalize((camera.viewInverse * focalPoint) - origin).xyz), 0.0);
    }

    // For simplicity, to do motion blur sample time in [0, 1] as start time and end time.
    float time = pc.batchRayTime;

    Ray ray;
    ray.origin    = origin.xyz;
    ray.direction = direction.xyz;
    ray.time      = time;
    return ray;
}

void main() {
    uvec2 pixel = gl_LaunchIDEXT.xy;

    uint rngState = initRNG(pc.sampleBatch, pixel, pc.resolution);

    const vec2 pixelCenter = vec2(gl_LaunchIDEXT.xy) + vec2(0.5);

    float sqrtSpp = sqrt(float(pc.samplesPerPixel));
    float recipSqrtSpp = 1.0 / sqrtSpp;
    float spp = int(sqrtSpp) * int(sqrtSpp); // In case pc.samplesPerPixel is not a perfect square.

    vec3 summedPixelColour = vec3(0.0);
    for (int sj = 0; sj < sqrtSpp; ++sj) {
        for (int si = 0; si < sqrtSpp; ++si) {
            Ray ray = getRay(rngState, pixelCenter, si, sj, recipSqrtSpp);
            vec3 attenuation = rayColour(rngState, ray, RAY_EPS, RAY_INF, gl_RayFlagsNoneEXT);
            summedPixelColour += attenuation;
        }
    }

    // Blend with the averaged image in the buffer:
    vec3 averagePixelColour = summedPixelColour / spp;
    if (pc.sampleBatch != 0) {
        vec3 imageData = imageLoad(image, ivec2(pixel)).rgb;
        averagePixelColour = (pc.sampleBatch * imageData + averagePixelColour) / (pc.sampleBatch + 1);
    }

    imageStore(image, ivec2(pixel), vec4(averagePixelColour, 1.0));
}
