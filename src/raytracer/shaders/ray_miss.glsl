#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 rayPayload;

void main() {
    rayPayload = vec3(0.0, 0.0, 0.2);
}
