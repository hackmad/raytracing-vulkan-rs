#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_scalar_block_layout: enable
#extension GL_EXT_nonuniform_qualifier : enable

#include "common.glsl"

layout(location = 0) rayPayloadInEXT vec4 rayPayload;
hitAttributeEXT vec2 hitAttribs;

layout(set = 3, binding = 0, scalar) buffer MeshData {
    Mesh values[];
} mesh_data;

layout(set = 4, binding = 0) uniform sampler texture_sampler;
layout(set = 4, binding = 1) uniform texture2D textures[];

Vertex unpackInstanceVertex(const int instanceId, const int primitiveId) {
    vec3 barycentricCoords = vec3(1.0 - hitAttribs.x - hitAttribs.y, hitAttribs.x, hitAttribs.y);

    Mesh mesh = mesh_data.values[instanceId];

    uint i = primitiveId * 3;
    uint i0 = mesh.indices.values[i];
    uint i1 = mesh.indices.values[i + 1];
    uint i2 = mesh.indices.values[i + 2];

    Vertex v0 = mesh.vertices.values[i0];
    Vertex v1 = mesh.vertices.values[i1];
    Vertex v2 = mesh.vertices.values[i2];

    const vec3 position =
        v0.position * barycentricCoords.x +
        v1.position * barycentricCoords.y +
        v2.position * barycentricCoords.z;

    const vec3 normal =
        v0.normal * barycentricCoords.x +
        v1.normal * barycentricCoords.y +
        v2.normal * barycentricCoords.z;

    const vec2 texCoord = 
        v0.texCoord * barycentricCoords.x +
        v1.texCoord * barycentricCoords.y +
        v2.texCoord * barycentricCoords.z;


    const vec3 worldSpacePosition = vec3(gl_ObjectToWorldEXT * vec4(position, 1.0));
    const vec3 worldSpaceNormal = normalize(vec3(normal * gl_WorldToObjectEXT));
    return Vertex(worldSpacePosition, worldSpaceNormal, texCoord);
}

void main() {
    Vertex vertex = unpackInstanceVertex(gl_InstanceID, gl_PrimitiveID);
    rayPayload = texture(nonuniformEXT(sampler2D(textures[0], texture_sampler)), vertex.texCoord);
}

