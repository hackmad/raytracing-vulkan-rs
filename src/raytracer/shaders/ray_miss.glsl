#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec4 rayPayload;

void main() {
    // For now just use a dark blue solid background color.
    rayPayload = vec4(0.0, 0.0, 0.2, 1.0);
}
