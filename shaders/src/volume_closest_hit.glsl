#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable

#include "common.glsl"

layout(location = 0) rayPayloadInEXT RayPayload rp;

void main() {
    rp.isMissed       = false;
    rp.isVolume       = true;
    rp.volumeIndex    = gl_InstanceCustomIndexEXT + gl_PrimitiveID;
    rp.tVolume        = gl_RayTmaxEXT;
    rp.volumeEntering = (gl_HitKindEXT == HIT_VOLUME_ENTER);
    rp.objectToWorld  = gl_ObjectToWorldEXT;
    rp.worldToObject  = gl_WorldToObjectEXT;
}
