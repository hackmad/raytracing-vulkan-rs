#version 460
#extension GL_EXT_ray_tracing : require

#include "common.glsl"

layout(location = 0) rayPayloadEXT RayPayload rayPayload;

layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;

layout(set = 1, binding = 0) uniform Camera {
    mat4 viewProj;    // Camera view * projection
    mat4 viewInverse; // Camera inverse view matrix
    mat4 projInverse; // Camera inverse projection matrix
} camera;

layout(set = 2, binding = 0, rgba8) uniform image2D image;

layout(push_constant) uniform PushConstantData {
    uvec2 resolution;
    uint  samplesPerPixel;
    uint  maxRayDepth;
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
    uint rngState = initRNG(gl_LaunchIDEXT.xy, pc.resolution);

    float pixelSampleScale = 1.0 / float(pc.samplesPerPixel);

    uint rayFlags = gl_RayFlagsOpaqueEXT;
    float tMin = 0.001;
    float tMax = 10000.0;

    const vec2 pixelCenter = vec2(gl_LaunchIDEXT.xy) + vec2(0.5);

    vec3 pixelColour = vec3(0.0);
    for (int i = 0; i < pc.samplesPerPixel; ++i) {
        const vec2 randomPixelCenter = pixelCenter + sampleSquare(rngState);

        const vec2 screenUV = randomPixelCenter / vec2(gl_LaunchSizeEXT.xy);
        vec2 d = screenUV * 2.0 - 1.0;

        vec4 origin = camera.viewInverse * vec4(0.0, 0.0, 0.0, 1.0);
        vec4 target = camera.projInverse * vec4(d.x, d.y, 1.0, 1.0);
        vec4 direction = camera.viewInverse * vec4(normalize(target.xyz), 0.0);

        vec3 attenuation = rayColour(rngState, origin, direction, tMin, tMax, rayFlags);
        pixelColour += pixelSampleScale * attenuation;
    }

    imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(linearTosRGB(pixelColour), 1.0));
}
