#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_scalar_block_layout: enable
#extension GL_EXT_nonuniform_qualifier : enable

#include "common.glsl"

layout(location = 0) rayPayloadInEXT vec4 rayPayload;
hitAttributeEXT vec2 hitAttribs;

layout(buffer_reference, scalar) readonly buffer Vertices {
    Vertex values[];
};
layout(buffer_reference, scalar) readonly buffer Indices {
    uvec3 values[];
};
layout(set = 3, binding = 0, scalar) readonly buffer Data {
    MeshData values[];
} data;

layout(set = 4, binding = 0) uniform sampler texture_sampler;
layout(set = 4, binding = 1) uniform texture2D textures[];

Vertex unpackInstanceVertex(const int intanceId) {
    vec3 barycentricCoords = vec3(1.0 - hitAttribs.x - hitAttribs.y, hitAttribs.x, hitAttribs.y);

    MeshData meshData = data.values[intanceId];

    Vertices vertices = Vertices(meshData.vertexBufferAddress);
    Indices indices = Indices(meshData.indexBufferAddress);

    uvec3 triangleIndices = indices.values[gl_PrimitiveID];

    Vertex v0 = vertices.values[triangleIndices.x];
    Vertex v1 = vertices.values[triangleIndices.y];
    Vertex v2 = vertices.values[triangleIndices.z];

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
    Vertex vertex = unpackInstanceVertex(gl_InstanceCustomIndexEXT);
    rayPayload = texture(nonuniformEXT(sampler2D(textures[0], texture_sampler)), vertex.texCoord);
}

