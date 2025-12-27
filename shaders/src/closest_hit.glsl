#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable

#include "common.glsl"
#include "perlin.glsl"

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
layout(set = 6, binding = 3, scalar) buffer DiffuseLightMaterials {
    DiffuseLightMaterial values[];
} diffuseLightMaterial;

layout(set = 7, binding = 0, scalar) buffer CheckerTextures {
    CheckerTexture values[];
} checkerTexture;
layout(set = 7, binding = 1, scalar) buffer NoiseTextures {
    NoiseTexture values[];
} noiseTexture;

layout(set = 9, binding = 0, scalar) buffer LightSourceAliasTable {
    LightSourceAliasTableEntry values[];
} lightSourceAliasTableData;

// If RayGenPushConstants in ray_gen.glsl changes, make sure to update layouts here.
layout(push_constant) uniform ClosestHitPushConstants {
    layout(offset = 20) uint  meshCount;
    layout(offset = 24) uint  imageTextureCount;
    layout(offset = 28) uint  constantColourCount;
    layout(offset = 32) uint  checkerTextureCount;
    layout(offset = 36) uint  noiseTextureCount;
    layout(offset = 40) uint  lambertianMaterialCount;
    layout(offset = 44) uint  metalMaterialCount;
    layout(offset = 48) uint  dielectricMaterialCount;
    layout(offset = 52) uint  diffuseLightMaterialCount;
    layout(offset = 56) uint  lightSourceTriangleCount;
    layout(offset = 60) float lightSourceTotalArea;
} pc;

struct MeshMaterial {
    uint type;
    uint index;
};

struct MeshTriangle {
    MeshVertex v0;
    MeshVertex v1;
    MeshVertex v2;
};

MeshMaterial unpackInstanceMaterial(const uint meshId) {
    Mesh mesh = meshData.values[meshId];
    return MeshMaterial(mesh.materialType, mesh.materialIndex);
}

MeshTriangle unpackInstanceVertex(const uint meshId, const uint primitiveId) {
    Mesh mesh = meshData.values[meshId];

    // Note if we got here meshId >= 1 and pc.meshCount >= 1 because there was an intersection.
    uint indexBufferOffset = 0;
    uint vertexBufferOffset = 0;
    for (uint id = 0; id < meshId && id < pc.meshCount; id++) {
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

    return MeshTriangle(v0, v1, v2);
}

HitRecord getIntersection(MeshTriangle tri) {
    vec3 barycentricCoords = vec3(1.0 - hitAttribs.x - hitAttribs.y, hitAttribs.x, hitAttribs.y);

    const vec3 position =
        tri.v0.p * barycentricCoords.x +
        tri.v1.p * barycentricCoords.y +
        tri.v2.p * barycentricCoords.z;

    const vec3 normal =
        tri.v0.n * barycentricCoords.x +
        tri.v1.n * barycentricCoords.y +
        tri.v2.n * barycentricCoords.z;

    const float u =
        tri.v0.u * barycentricCoords.x +
        tri.v1.u * barycentricCoords.y +
        tri.v2.u * barycentricCoords.z;

    const float v =
        tri.v0.v * barycentricCoords.x +
        tri.v1.v * barycentricCoords.y +
        tri.v2.v * barycentricCoords.z;

    const vec3 worldSpacePosition = vec3(gl_ObjectToWorldEXT * vec4(position, 1.0));
    const vec3 worldSpaceNormal = normalize(vec3(normal * gl_WorldToObjectEXT));

    bool frontFace = isFrontFace(gl_WorldRayDirectionEXT, worldSpaceNormal);

    return HitRecord(
        MeshVertex(worldSpacePosition, u, worldSpaceNormal, v),
        frontFace,
        frontFace ? worldSpaceNormal : -worldSpaceNormal
    );
}

// This only handles constant colour, image and noise textures. Other textures like checker texture can reference
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

        case MAT_PROP_VALUE_TYPE_NOISE:
            if (matPropValue.index >= 0 && matPropValue.index < pc.noiseTextureCount) {
                float scale = noiseTexture.values[matPropValue.index].scale;
                colour = vec3(0.5, 0.5, 0.5) * (1.0 + sin(scale * vertex.p.z + 10 * turbulence(vertex.p, 7)));
            }
            break;
    }

    return colour;
}

vec3 getMaterialPropertyValue(MaterialPropertyValue matPropValue, MeshVertex vertex) {
    vec3 colour = vec3(0.0);

    switch (matPropValue.propValueType) {
        case MAT_PROP_VALUE_TYPE_RGB:
        case MAT_PROP_VALUE_TYPE_IMAGE:
        case MAT_PROP_VALUE_TYPE_NOISE:
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

LightSample sampleLightSource(inout uint rngState) {
    float u1 = randomFloat(rngState);
    float u2 = randomFloat(rngState);

    uint i = min(uint(u1 * pc.lightSourceTriangleCount), pc.lightSourceTriangleCount - 1);

    uint triangleIndex;
    if (u2 < lightSourceAliasTableData.values[i].probability) {
        triangleIndex = i;
    } else {
        triangleIndex = lightSourceAliasTableData.values[i].alias;
    }

    uint meshId        = lightSourceAliasTableData.values[triangleIndex].meshId;
    uint primitiveId   = lightSourceAliasTableData.values[triangleIndex].primitiveId;
    MeshTriangle light = unpackInstanceVertex(meshId, primitiveId);

    vec3 position = sampleTriangleUniform(rngState, light.v0.p, light.v1.p, light.v2.p);
    vec3 normal   = normalize(cross(light.v1.p - light.v0.p, light.v2.p - light.v0.p));
    float pdf     = 1.0 / pc.lightSourceTotalArea;

    return LightSample(position, normal, pdf);
}


void lambertianMaterialScatter(uint materialIndex, HitRecord rec) {
    if (materialIndex >= 0 && materialIndex < pc.lambertianMaterialCount) {
        LambertianMaterial material = lambertianMaterial.values[materialIndex];
        vec3 albedo = getMaterialPropertyValue(material.albedo, rec.meshVertex);

        // Calculate cosine PDF.
        ONB uvw = createOrthonormalBases(rec.normal);
        vec3 scatterDirection = onbTransform(uvw, randomVec3CosineDirection(rayPayload.rngState));

        float cosTheta = dot(rec.normal, normalize(scatterDirection));
        float scatteringPdf = max(0.0, cosTheta / PI);

        float pdf = dot(uvw.axis[2], scatterDirection) / PI;

        rayPayload.scatterColour = scatteringPdf * albedo / pdf;
        rayPayload.isScattered = true;
        rayPayload.scatteredRay.direction = scatterDirection;
        rayPayload.scatteredRay.origin = rec.meshVertex.p;
    }
}

void metalMaterialScatter(uint materialIndex, HitRecord rec) {
    if (materialIndex >= 0 && materialIndex < pc.metalMaterialCount) {
        MetalMaterial material = metalMaterial.values[materialIndex];
        vec3 albedo = getMaterialPropertyValue(material.albedo, rec.meshVertex);
        vec3 fuzz = getMaterialPropertyValue(material.fuzz, rec.meshVertex);

        vec3 reflectedDirection = reflect(gl_WorldRayDirectionEXT, rec.normal);

        vec3 scatteredDirection = normalize(reflectedDirection) +
            (fuzz * randomUnitVec3(rayPayload.rngState));

        rayPayload.scatterColour = albedo;
        rayPayload.isScattered = (dot(scatteredDirection, rec.normal) > 0);
        rayPayload.scatteredRay.direction = scatteredDirection;
        rayPayload.scatteredRay.origin = rec.meshVertex.p;
    }
}

void dielectricMaterialScatter(uint materialIndex, HitRecord rec) {
    if (materialIndex >= 0 && materialIndex < pc.dielectricMaterialCount) {
        DielectricMaterial material = dielectricMaterial.values[materialIndex];
        float refractionIndex = material.refractionIndex;

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

        rayPayload.scatterColour = attenuation;
        rayPayload.isScattered = true;
        rayPayload.scatteredRay.direction = refractedDirection;
        rayPayload.scatteredRay.origin = rec.meshVertex.p;
    }
}

void diffuseLightMaterialEmission(uint materialIndex, HitRecord rec) {
    if (materialIndex >= 0 && materialIndex < pc.diffuseLightMaterialCount) {
        DiffuseLightMaterial material = diffuseLightMaterial.values[materialIndex];
        if (rec.isFrontFace) {
            rayPayload.emissionColour = getMaterialPropertyValue(material.emit, rec.meshVertex);
        }
    }
}

void calculateScatter(MeshMaterial material, HitRecord rec) {
    switch (material.type) {
        case MAT_TYPE_LAMBERTIAN:
            lambertianMaterialScatter(material.index, rec);
            break;

        case MAT_TYPE_METAL:
            metalMaterialScatter(material.index, rec);
            break;

        case MAT_TYPE_DIELECTRIC:
            dielectricMaterialScatter(material.index, rec);
            break;

        default:
            // Materials that don't support scattering.
            rayPayload.isScattered = false;
            rayPayload.scatterColour = vec3(0.0);
            break;
    }
}

void calculateEmission(MeshMaterial material, HitRecord rec) {
    switch (material.type) {
        case MAT_TYPE_DIFFUSE_LIGHT:
            diffuseLightMaterialEmission(material.index, rec);
            break;

        default:
            // Non-emissive materials.
            rayPayload.emissionColour = vec3(0.0);
            break;
    }
}

void main() {
    MeshTriangle tri = unpackInstanceVertex(gl_InstanceCustomIndexEXT, gl_PrimitiveID);
    HitRecord rec = getIntersection(tri);
    MeshMaterial material = unpackInstanceMaterial(gl_InstanceCustomIndexEXT);

    calculateScatter(material, rec);
    calculateEmission(material, rec);
}

