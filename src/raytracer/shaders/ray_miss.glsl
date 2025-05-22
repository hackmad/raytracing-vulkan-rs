#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 rayPayload;

void main() {
    // For now just use a dark blue solid background colour.
    rayPayload = vec3(0.0, 0.0, 0.2);
}
