#version 460
#extension GL_EXT_ray_tracing : require

#include "common.glsl"

layout(location = 0) rayPayloadEXT vec3 rayPayload;

layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;

layout(set = 1, binding = 0) uniform Camera {
    mat4 viewProj;    // Camera view * projection
    mat4 viewInverse; // Camera inverse view matrix
    mat4 projInverse; // Camera inverse projection matrix
} camera;

layout(set = 2, binding = 0, rgba8) uniform image2D image;

void main() {
    const vec2 pixelCenter = vec2(gl_LaunchIDEXT.xy) + vec2(0.5);
    const vec2 in_uv = pixelCenter / vec2(gl_LaunchSizeEXT.xy);
    vec2 d = in_uv * 2.0 - 1.0;

    vec4 origin = camera.viewInverse * vec4(0, 0, 0, 1);
    vec4 target = camera.projInverse * vec4(d.x, d.y, 1, 1);
    vec4 direction = camera.viewInverse * vec4(normalize(target.xyz), 0);

    uint rayFlags = gl_RayFlagsOpaqueEXT;
    float tMin = 0.001;
    float tMax = 10000.0;

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

    vec3 color = linearTosRGB(rayPayload);
    imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(color, 1.0));
}
