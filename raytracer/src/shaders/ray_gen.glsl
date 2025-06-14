#version 460
#extension GL_EXT_ray_tracing : require

#include "common.glsl"

layout(location = 0) rayPayloadEXT RayPayload rayPayload;

layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;

layout(set = 1, binding = 0) uniform Camera {
    mat4  viewProj;     // Camera view * projection
    mat4  viewInverse;  // Camera inverse view matrix
    mat4  projInverse;  // Camera inverse projection matrix
    float focalLength;  // Focal length of lens.
    float apertureSize; // Aperture size (diameter of lens).
} camera;

layout(set = 2, binding = 0, rgba8) uniform image2D image;

layout(push_constant) uniform RayGenPushConstants {
    layout(offset = 16) uvec2 resolution;
    uint samplesPerPixel; // Don't exceed 64. See https://nvpro-samples.github.io/vk_mini_path_tracer/extras.html#moresamples.
    uint sampleBatches;   // Don't exceed 32.
    uint sampleBatch;
    uint maxRayDepth;
} pc;

vec3 rayColour(inout uint rngState, vec4 origin, vec4 direction, float tMin, float tMax, uint rayFlags) {
    uint rayDepth = 0;
    vec3 attenuation = vec3(1.0);

    while (rayDepth < pc.maxRayDepth) {
        // Initialize ray payload with RNG state. It will get used in different shaders like closest_hit.glsl.
        // So store updated RNG state after ray traversal completes.
        rayPayload = initRayPayload(rngState);

        // sbtRecordOffset, sbtRecordStride control how the hitGroupId (VkAccelerationStructureInstanceKHR::
        // instanceShaderBindingTableRecordOffset) of each instance is used to look up a hit group in the 
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
                origin.xyz,    // ray origin
                tMin,          // ray min range
                direction.xyz, // ray direction
                tMax,          // ray max range
                0);            // payload (location = 0)

        rngState = rayPayload.rngState;

        // Closest hit and miss shader will set rayPayload fields.
        if (!rayPayload.isMissed) {
            attenuation *= rayPayload.attenuation;

            if (!rayPayload.isScattered) break;
        } else {
            vec3 unitDirection = normalize(direction.xyz);
            float a = 0.5 * (unitDirection.y + 1.0);
            attenuation *= mix(vec3(1.0, 1.0, 1.0), vec3(0.5, 0.7, 1.0), a);
            break;
        } 

        origin = vec4(rayPayload.scatteredRayOrigin, 1.0);
        direction = vec4(rayPayload.scatteredRayDirection, 1.0);
        rayDepth++;
    }

    return attenuation;
}

void main() {
    uvec2 pixel = gl_LaunchIDEXT.xy;

    uint rngState = initRNG(pc.sampleBatch, pixel, pc.resolution);

    uint rayFlags = gl_RayFlagsOpaqueEXT;
    float tMin = 0.001;
    float tMax = 10000.0;

    const vec2 pixelCenter = vec2(gl_LaunchIDEXT.xy) + vec2(0.5);

    vec3 summedPixelColour = vec3(0.0);
    for (int i = 0; i < pc.samplesPerPixel; ++i) {
        const vec2 randomPixelCenter = pixelCenter + (0.375 * randomGaussian(rngState)); // + sampleSquare(rngState);

        const vec2 screenUV = randomPixelCenter / vec2(gl_LaunchSizeEXT.xy);
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

        vec3 attenuation = rayColour(rngState, origin, direction, tMin, tMax, rayFlags);
        summedPixelColour += attenuation;
    }

    // Blend with the averaged image in the buffer:
    vec3 averagePixelColour = summedPixelColour / pc.samplesPerPixel;
    if (pc.sampleBatch != 0) {
        vec3 imageData = sRGBToLinear(imageLoad(image, ivec2(pixel)).xyz);
        averagePixelColour = (pc.sampleBatch * imageData + averagePixelColour) / (pc.sampleBatch + 1);
    }

    imageStore(image, ivec2(pixel), vec4(linearTosRGB(averagePixelColour), 1.0));
}
