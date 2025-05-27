#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 rayPayload;

void main() {
    vec3 unitDirection = normalize(gl_WorldRayDirectionEXT);
    float a = 0.5 * (unitDirection.y + 1.0);
    rayPayload = mix(vec3(1.0, 1.0, 1.0), vec3(0.5, 0.7, 1.0), a);
}
