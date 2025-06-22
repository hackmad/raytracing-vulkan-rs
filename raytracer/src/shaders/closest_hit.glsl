#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable

#include "common.glsl"

layout(location = 0) rayPayloadInEXT RayPayload rayPayload;
hitAttributeEXT vec2 hitAttribs;

layout(location = 1) rayPayloadEXT bool isShadowed;

layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;

layout(set = 3, binding = 0, scalar) buffer MeshVertices {
    MeshVertex values[];
} meshVertexData;

layout(set = 3, binding = 1, scalar) buffer MeshIndices {
    uint values[];
} meshIndexData;

layout(set = 3, binding = 2, scalar) buffer Meshes {
    Mesh values[];
} meshData;

layout(set = 4, binding = 0) uniform sampler imageTextureSampler;
layout(set = 4, binding = 1) uniform texture2D imageTextures[];

layout(set = 5, binding = 0, scalar) buffer ConstantColours {
    vec3 values[];
} constantColour;

layout(set = 6, binding = 0, scalar) buffer LambertianMaterials {
    LambertianMaterial values[];
} lambertianMaterial;
layout(set = 6, binding = 1, scalar) buffer MetalMaterials {
    MetalMaterial values[];
} metalMaterial;
layout(set = 6, binding = 2, scalar) buffer DielectricMaterials {
    DielectricMaterial values[];
} dielectricMaterial;

layout(set = 7, binding = 0, scalar) buffer CheckerTextures {
    CheckerTexture values[];
} checkerTexture;

layout(push_constant) uniform ClosestHitPushConstants {
    uint meshCount;
    uint imageTextureCount;
    uint constantColourCount;
    uint checkerTextureCount;
    uint lambertianMaterialCount;
    uint metalMaterialCount;
    uint dielectricMaterialCount;
} pc;

HitRecord unpackInstanceVertex(const int meshInstanceId, const int primitiveId) {
    vec3 barycentricCoords = vec3(1.0 - hitAttribs.x - hitAttribs.y, hitAttribs.x, hitAttribs.y);

    Mesh mesh = meshData.values[meshInstanceId];

    // Note if we got here meshInstanceId >= 1 and pc.meshCount >= 1 because there was an intersection.
    uint indexBufferOffset = 0;
    uint vertexBufferOffset = 0;
    for (uint id = 0; id < meshInstanceId && id < pc.meshCount; id++) {
        indexBufferOffset += meshData.values[id].indexBufferSize;
        vertexBufferOffset += meshData.values[id].vertexBufferSize;
    }

    uint i = indexBufferOffset + primitiveId * 3;
    uint i0 = meshIndexData.values[i];
    uint i1 = meshIndexData.values[i + 1];
    uint i2 = meshIndexData.values[i + 2];

    MeshVertex v0 = meshVertexData.values[vertexBufferOffset + i0];
    MeshVertex v1 = meshVertexData.values[vertexBufferOffset + i1];
    MeshVertex v2 = meshVertexData.values[vertexBufferOffset + i2];

    const vec3 position =
        v0.p * barycentricCoords.x +
        v1.p * barycentricCoords.y +
        v2.p * barycentricCoords.z;

    const vec3 normal =
        v0.n * barycentricCoords.x +
        v1.n * barycentricCoords.y +
        v2.n * barycentricCoords.z;

    const float u = 
        v0.u * barycentricCoords.x +
        v1.u * barycentricCoords.y +
        v2.u * barycentricCoords.z;

    const float v = 
        v0.v * barycentricCoords.x +
        v1.v * barycentricCoords.y +
        v2.v * barycentricCoords.z;

    const vec3 worldSpacePosition = vec3(gl_ObjectToWorldEXT * vec4(position, 1.0));
    const vec3 worldSpaceNormal = normalize(vec3(normal * gl_WorldToObjectEXT));

    bool frontFace = isFrontFace(gl_WorldRayDirectionEXT, worldSpaceNormal);

    return HitRecord(
            MeshVertex(worldSpacePosition, u, worldSpaceNormal, v),
            frontFace,
            frontFace ? worldSpaceNormal : -worldSpaceNormal
    );
}

// This only handles constant colour and image textures. Other textures like checker texture can reference
// these "basic" textures for their own properties.
vec3 getBasicTextureValue(MaterialPropertyValue matPropValue, MeshVertex vertex) {
    vec3 colour = vec3(0.0);

    switch (matPropValue.propValueType) {
        case MAT_PROP_VALUE_TYPE_RGB:
            if (matPropValue.index >= 0 && matPropValue.index < pc.constantColourCount) {
                colour = constantColour.values[matPropValue.index];
            }
            break;

        case MAT_PROP_VALUE_TYPE_IMAGE:
            if (matPropValue.index >= 0 && matPropValue.index < pc.imageTextureCount) {
                colour = texture(
                        nonuniformEXT(sampler2D(imageTextures[matPropValue.index], imageTextureSampler)),
                        vec2(vertex.u, vertex.v)
                        ).rgb; // Ignore alpha for now.
            }
            break;
    }

    return colour;
}

vec3 getMaterialPropertyValue(MaterialPropertyValue matPropValue, MeshVertex vertex) {
    vec3 colour = vec3(0.0);

    switch (matPropValue.propValueType) {
        case MAT_PROP_VALUE_TYPE_RGB:
            colour = getBasicTextureValue(matPropValue, vertex);
            break;

        case MAT_PROP_VALUE_TYPE_IMAGE:
            colour = getBasicTextureValue(matPropValue, vertex);
            break;

        case MAT_PROP_VALUE_TYPE_CHECKER:
            if (matPropValue.index >= 0 && matPropValue.index < pc.checkerTextureCount) {
                CheckerTexture texture = checkerTexture.values[matPropValue.index];

                float invScale = 1.0 / texture.scale;
                int xInteger = int(floor(invScale * vertex.p.x));
                int yInteger = int(floor(invScale * vertex.p.y));
                int zInteger = int(floor(invScale * vertex.p.z));

                bool isEven = (xInteger + yInteger + zInteger) % 2 == 0;

                colour = isEven 
                    ? getBasicTextureValue(texture.even, vertex)
                    : getBasicTextureValue(texture.odd, vertex);
                
            }
            break;
    }

    return colour;
}

// Use Schlick's approximation for reflectance.
float schlickReflectance(float cosine, float refractionIndex) {
    float r0 = (1.0 - refractionIndex) / (1.0 + refractionIndex);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow((1.0 - cosine), 5);
}

void scatterLambertianMaterial(uint materialIndex, HitRecord rec) {
    if (materialIndex >= 0 && materialIndex < pc.lambertianMaterialCount) {
        LambertianMaterial material = lambertianMaterial.values[materialIndex];
        vec3 albedo = getMaterialPropertyValue(material.albedo, rec.meshVertex);

        vec3 scatterDirection = rec.normal + randomUnitVec3(rayPayload.rngState);

        // Catch degenerate scatter direction.
        if (nearZero(scatterDirection)) {
            scatterDirection = rec.normal;
        }

        rayPayload.attenuation = albedo;
        rayPayload.isScattered = true;
        rayPayload.scatteredRayDirection = scatterDirection;
        rayPayload.scatteredRayOrigin = rec.meshVertex.p;
    }
}

void scatterMetalMaterial(uint materialIndex, HitRecord rec) {
    if (materialIndex >= 0 && materialIndex < pc.metalMaterialCount) {
        MetalMaterial material = metalMaterial.values[materialIndex];
        vec3 albedo = getMaterialPropertyValue(material.albedo, rec.meshVertex);
        vec3 fuzz = getMaterialPropertyValue(material.fuzz, rec.meshVertex);

        vec3 reflectedDirection = reflect(gl_WorldRayDirectionEXT, rec.normal);

        vec3 scatteredDirection = normalize(reflectedDirection) +
            (fuzz * randomUnitVec3(rayPayload.rngState));

        rayPayload.attenuation = albedo;
        rayPayload.isScattered = (dot(scatteredDirection, rec.normal) > 0);
        rayPayload.scatteredRayDirection = scatteredDirection;
        rayPayload.scatteredRayOrigin = rec.meshVertex.p;
    }
}

void scatterDielectricMaterial(uint materialIndex, HitRecord rec) {
    if (materialIndex >= 0 && materialIndex < pc.dielectricMaterialCount) {
        DielectricMaterial dielectricMaterial = dielectricMaterial.values[materialIndex];
        float refractionIndex = dielectricMaterial.refractionIndex;

        vec3 attenuation = vec3(1.0);

        float ri = rec.isFrontFace ? (1.0 / refractionIndex) : refractionIndex;

        vec3 unitDirection = normalize(gl_WorldRayDirectionEXT);

        float cosTheta = min(dot(-unitDirection, rec.normal), 1.0);
        float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

        bool cannotRefract = ri * sinTheta > 1.0; 
        cannotRefract = cannotRefract || schlickReflectance(cosTheta, ri) > randomFloat(rayPayload.rngState);

        vec3 refractedDirection = cannotRefract
            ? reflect(unitDirection, rec.normal) // Total internal reflection.
            : refract(unitDirection, rec.normal, ri);

        rayPayload.attenuation = attenuation;
        rayPayload.isScattered = true;
        rayPayload.scatteredRayDirection = refractedDirection;
        rayPayload.scatteredRayOrigin = rec.meshVertex.p;
    }
}

void unpackInstanceMaterialAndScatter(const int meshInstanceId, HitRecord rec) {
    Mesh mesh = meshData.values[meshInstanceId];
    uint materialType = mesh.materialType;
    uint materialIndex = mesh.materialIndex;

    switch (materialType) {
        case MAT_TYPE_LAMBERTIAN:
            scatterLambertianMaterial(materialIndex, rec);
            break;

        case MAT_TYPE_METAL:
            scatterMetalMaterial(materialIndex, rec);
            break;

        case MAT_TYPE_DIELECTRIC:
            scatterDielectricMaterial(materialIndex, rec);
            break;
    }
}

void main() {
    HitRecord rec = unpackInstanceVertex(gl_InstanceID, gl_PrimitiveID);
    unpackInstanceMaterialAndScatter(gl_InstanceID, rec);
}

