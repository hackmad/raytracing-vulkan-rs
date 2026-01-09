#version 460
#extension GL_EXT_ray_tracing : require

#include "common.glsl"

hitAttributeEXT VolumeHitAttribs hitAttribs;

// Volumes will hold AABBs for all volumes. For box volumes it holds all necessary data for box shaped volumes; so we 
// don't need a separate BoxVolumes binding.
layout(set = 10, binding = 0, scalar) buffer Volumes {
    Volume values[];
} volumeData;
layout(set = 10, binding = 1, scalar) buffer SphereVolumes {
    SphereVolume values[];
} sphereVolumeData;

// Make sure to check layout offsets for push constants in each of the shader files.
// The order of these matter so make sure they are consistent across all shaders.
layout(push_constant) uniform PushConstants {
    layout(offset = 68) uint volumeCount;
} pc;

void main() {
    uint volumeIndex = gl_InstanceCustomIndexEXT + gl_PrimitiveID;
    Volume volume = volumeData.values[volumeIndex];

    float tMin;
    float tMax;
    bool  isHit;
    switch (volume.shape) {
        case VOLUME_SHAPE_BOX:
            isHit = rayIntersectBox(gl_ObjectRayOriginEXT, gl_ObjectRayDirectionEXT,
                    volume.aabbMin, volume.aabbMax, tMin, tMax);
            break;

        case VOLUME_SHAPE_SPHERE:
            SphereVolume sphereVolume = sphereVolumeData.values[volume.shapeVolumeIndex];
            isHit = rayIntersectSphere(gl_ObjectRayOriginEXT, gl_ObjectRayDirectionEXT,
                    sphereVolume.center, sphereVolume.radius, tMin, tMax);
            break;

        default:
            isHit = false;
            break;
    }

    if (!isHit) return;

    // Clamp to ray interval.
    tMin = max(tMin, gl_RayTminEXT);
    tMax = min(tMax, gl_RayTmaxEXT);
    if (tMin >= tMax) return;

    hitAttribs.volumeIndex = volumeIndex;
    hitAttribs.tEntry = tMin;
    hitAttribs.tExit  = tMax;
    reportIntersectionEXT(tMin, 0);
}

