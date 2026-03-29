#version 460
#extension GL_EXT_ray_tracing : require

#include "common.glsl"

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
    layout(offset = 76) uint volumeCount;
} pc;

void main() {
    uint volumeIndex = gl_InstanceCustomIndexEXT + gl_PrimitiveID;
    if (volumeIndex >= pc.volumeCount) {
        return;
    }

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

    if (!isHit || tMax < gl_RayTminEXT || tMin > gl_RayTmaxEXT || tMin >= tMax) {
        return;
    }

    float tHit;
    uint  hitKind;
    if (tMin >= gl_RayTminEXT) {
        tHit = tMin;
        hitKind = HIT_VOLUME_ENTER;
    } else {
        tHit = tMax;
        hitKind = HIT_VOLUME_EXIT;
    }

    reportIntersectionEXT(tHit, hitKind);
}

