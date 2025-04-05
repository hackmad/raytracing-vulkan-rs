#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_scalar_block_layout: enable

#include "common.glsl"

layout(location = 0) rayPayloadInEXT vec3 rayPayload;
hitAttributeEXT vec2 hitAttribs;

layout(buffer_reference, scalar) buffer Vertices {
    Vertex values[];
};
layout(buffer_reference, scalar) buffer Indices {
    uvec3 values[];
};
layout(set = 3, binding = 0, scalar) buffer Data {
    MeshData values[];
} data;

Vertex unpackInstanceVertex(const int intanceId) {
    MeshData meshData = data.values[intanceId];
    Vertices vertices = Vertices(meshData.vertexBufferAddress);
    Indices indices = Indices(meshData.indexBufferAddress);

    uvec3 triangleIndices = indices.values[gl_PrimitiveID];
    Vertex v0 = vertices.values[triangleIndices.x];
    Vertex v1 = vertices.values[triangleIndices.y];
    Vertex v2 = vertices.values[triangleIndices.z];

    vec3 barycentricCoords = vec3(1.0 - hitAttribs.x - hitAttribs.y, hitAttribs.x, hitAttribs.y);

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
    const Vertex vertex = unpackInstanceVertex(gl_InstanceCustomIndexEXT);
    rayPayload = map(vertex.normal, -1.0, 1.0, 0.0, 1.0);
}

