#version 460
#extension GL_EXT_ray_tracing : require

#include "common.glsl"

layout(location = 0) rayPayloadInEXT RayPayload rp;

void main() {
    rp.isMissed = true;
}
