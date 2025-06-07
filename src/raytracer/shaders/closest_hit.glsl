#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable

#include "common.glsl"

layout(location = 0) rayPayloadInEXT RayPayload rayPayload;
hitAttributeEXT vec2 hitAttribs;

layout(location = 1) rayPayloadEXT bool isShadowed;

layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;

layout(set = 3, binding = 0, scalar) buffer MeshData {
    Mesh values[];
} mesh_data;

layout(set = 4, binding = 0) uniform sampler textureSampler;
layout(set = 4, binding = 1) uniform texture2D textures[];

layout(set = 5, binding = 0) buffer MaterialColours {
    vec3 values[];
} materialColour;

layout(push_constant) uniform PushConstantData {
    uint textureCount;
    uint materialColourCount;
} pc;

MeshVertex unpackInstanceVertex(const int instanceId, const int primitiveId) {
    vec3 barycentricCoords = vec3(1.0 - hitAttribs.x - hitAttribs.y, hitAttribs.x, hitAttribs.y);

    Mesh mesh = mesh_data.values[instanceId];

    uint i = primitiveId * 3;
    uint i0 = mesh.indicesRef.values[i];
    uint i1 = mesh.indicesRef.values[i + 1];
    uint i2 = mesh.indicesRef.values[i + 2];

    MeshVertex v0 = mesh.verticesRef.values[i0];
    MeshVertex v1 = mesh.verticesRef.values[i1];
    MeshVertex v2 = mesh.verticesRef.values[i2];

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
    return MeshVertex(worldSpacePosition, worldSpaceNormal, texCoord);
}

vec3 unpackInstanceMaterial(const int instanceId, const uint matPropType, MeshVertex vertex) {
    Mesh mesh = mesh_data.values[instanceId];
    MaterialPropertyValue mat = mesh.materialsRef.values[matPropType];

    vec3 colour = vec3(0.0);
    switch (mat.propValueType) {
        case MAT_PROP_VALUE_TYPE_RGB:
            if (mat.index >= 0 && mat.index < pc.materialColourCount) {
                colour = materialColour.values[mat.index];
            }
            break;

        case MAT_PROP_VALUE_TYPE_TEXTURE:
            if (mat.index >= 0 && mat.index < pc.textureCount) {
                colour = texture(
                        nonuniformEXT(sampler2D(textures[mat.index], textureSampler)),
                        vertex.texCoord
                        ).rgb; // Ignore alpha for now.
            }
            break;
    }

    return colour;
}

void main() {
    MeshVertex vertex = unpackInstanceVertex(gl_InstanceID, gl_PrimitiveID);

    vec3 albedo = unpackInstanceMaterial(gl_InstanceID, MAT_PROP_TYPE_DIFFUSE, vertex);

    vec3 scatterDirection = vertex.normal + randomUnitVec3(rayPayload.rngState);

    // Catch degenerate scatter direction.
    if (nearZero(scatterDirection)) {
        scatterDirection = vertex.normal;
    }

    rayPayload.attenuation = albedo;
    rayPayload.isScattered = true;
    rayPayload.scatteredRayDirection = scatterDirection;
    rayPayload.scatteredRayOrigin = vertex.position;
}

