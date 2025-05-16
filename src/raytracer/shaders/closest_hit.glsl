#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout: enable
#extension GL_EXT_nonuniform_qualifier : enable

#include "common.glsl"

layout(location = 0) rayPayloadInEXT vec3 rayPayload;
hitAttributeEXT vec2 hitAttribs;

layout(set = 3, binding = 0, scalar) buffer MeshData {
    Mesh values[];
} mesh_data;

layout(set = 4, binding = 0) uniform sampler textureSampler;
layout(set = 4, binding = 1) uniform texture2D textures[];

layout(set = 5, binding = 0) buffer MaterialColors {
    vec3 values[];
} material_color;

layout(push_constant) uniform PushConstantData {
    uint texture_count;
    uint material_color_count;
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

Material unpackInstanceMaterial(const int instanceId, const uint mat_prop_type) {
    Mesh mesh = mesh_data.values[instanceId];
    return mesh.materialsRef.values[mat_prop_type];
}

void main() {
    MeshVertex vertex = unpackInstanceVertex(gl_InstanceID, gl_PrimitiveID);

    Material diffuse = unpackInstanceMaterial(gl_InstanceID, MAT_PROP_TYPE_DIFFUSE);

    // Diffuse color.
    vec3 diffuseColor = vec3(0.0, 0.0, 0.0);
    if (diffuse.propValueType == MAT_PROP_VALUE_TYPE_RGB) {
        if (diffuse.index >= 0 && diffuse.index < pc.material_color_count) {
            diffuseColor = material_color.values[diffuse.index];
        }
    } else if (diffuse.propValueType == MAT_PROP_VALUE_TYPE_TEXTURE) {
        if (diffuse.index >= 0 && diffuse.index < pc.texture_count) {
            diffuseColor = texture(
                nonuniformEXT(sampler2D(textures[diffuse.index], textureSampler)),
                vertex.texCoord
            ).rgb; // Ignore alpha for now.
        }
    }

    // TODO Use uniform buffers to pass these in.
    vec3 lightDir = vec3(1.0, 1.0, 0.0);
    vec3 ambientColor = vec3(0.05, 0.05, 0.05); // ambient term

    vec3 radiance = ambientColor;
    float irradiance = max(dot(lightDir, vertex.normal), 0.0);
    // TODO Check for shadows.
    if (irradiance > 0.0) { // if receives light
        radiance += diffuseColor * irradiance; // diffuse shading
    }

    rayPayload = radiance;
}

