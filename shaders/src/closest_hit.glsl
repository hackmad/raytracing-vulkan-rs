#version 460
#extension GL_EXT_ray_tracing : require

#include "common.glsl"

layout(location = 0) rayPayloadInEXT RayPayload rayPayload;
hitAttributeEXT vec2 hitAttribs;

void main() {
    rayPayload.meshId      = gl_InstanceCustomIndexEXT;
    rayPayload.primitiveId = gl_PrimitiveID;

    rayPayload.isMissed   = false;
    rayPayload.hitAttribs = hitAttribs;

    rayPayload.objectToWorld     = gl_ObjectToWorldEXT;
    rayPayload.worldToObject     = gl_WorldToObjectEXT;
    rayPayload.worldRayDirection = gl_WorldRayDirectionEXT;
}

