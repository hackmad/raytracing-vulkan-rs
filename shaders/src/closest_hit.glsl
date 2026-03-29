#version 460
#extension GL_EXT_ray_tracing : require

#include "common.glsl"

layout(location = 0) rayPayloadInEXT RayPayload rp;
hitAttributeEXT vec2 hitAttribs;

void main() {
    rp.isMissed          = false;
    rp.isSurface         = true;
    rp.t                 = gl_HitTEXT;
    rp.surfaceHitAttribs = hitAttribs;
    rp.meshId            = gl_InstanceCustomIndexEXT;
    rp.primitiveId       = gl_PrimitiveID;
    rp.objectToWorld     = gl_ObjectToWorldEXT;
    rp.worldToObject     = gl_WorldToObjectEXT;
}

