#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable

#include "common.glsl"
#include "perlin.glsl"

layout(location = 0) rayPayloadInEXT RayPayload rp;
hitAttributeEXT vec2 hitAttribs;

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

// Make sure to check layout offsets for push constants in each of the shader files.
// The order of these matter so make sure they are consistent across all shaders.
layout(push_constant) uniform PushConstants {
    layout(offset = 24) uint  meshCount;
    layout(offset = 28) uint  imageTextureCount;
    layout(offset = 32) uint  constantColourCount;
    layout(offset = 36) uint  checkerTextureCount;
    layout(offset = 40) uint  noiseTextureCount;
    layout(offset = 44) uint  lambertianMaterialCount;
    layout(offset = 48) uint  metalMaterialCount;
    layout(offset = 52) uint  dielectricMaterialCount;
    layout(offset = 56) uint  diffuseLightMaterialCount;
    layout(offset = 60) uint  lightSourceTriangleCount;
    layout(offset = 64) float lightSourceTotalArea;
} pc;

struct MeshTriangle {
    MeshVertex v0;
    MeshVertex v1;
    MeshVertex v2;
};

Material unpackInstanceMaterial(const uint meshId) {
    Mesh mesh = meshData.values[meshId];
    return Material(mesh.materialType, mesh.materialIndex);
}

MeshTriangle unpackInstanceVertex(const uint meshId, const uint primitiveId) {
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

HitRecord getIntersection(
        MeshTriangle hitTriangle,
        vec2         hitAttribs,
        mat4x3       objectToWorld,
        mat4x3       worldToObject,
        vec3         worldRayDirection) {
    vec3 barycentricCoords = vec3(1.0 - hitAttribs.x - hitAttribs.y, hitAttribs.x, hitAttribs.y);

    const vec3 position =
        hitTriangle.v0.p * barycentricCoords.x +
        hitTriangle.v1.p * barycentricCoords.y +
        hitTriangle.v2.p * barycentricCoords.z;

    const vec3 normal =
        hitTriangle.v0.n * barycentricCoords.x +
        hitTriangle.v1.n * barycentricCoords.y +
        hitTriangle.v2.n * barycentricCoords.z;

    const float u =
        hitTriangle.v0.u * barycentricCoords.x +
        hitTriangle.v1.u * barycentricCoords.y +
        hitTriangle.v2.u * barycentricCoords.z;

    const float v =
        hitTriangle.v0.v * barycentricCoords.x +
        hitTriangle.v1.v * barycentricCoords.y +
        hitTriangle.v2.v * barycentricCoords.z;

    const vec3 worldSpacePosition = vec3(objectToWorld * vec4(position, 1.0));
    const vec3 worldSpaceNormal = normalize(vec3(normal * worldToObject));

    bool frontFace = isFrontFace(worldRayDirection, worldSpaceNormal);

    return HitRecord(
        MeshVertex(worldSpacePosition, u, worldSpaceNormal, v),
        frontFace,
        frontFace ? worldSpaceNormal : -worldSpaceNormal
    );
}

// NOTE: getBasicTextureValue() and getMaterialPropertyValue() are duplicated from closest_hit. Would be nice to
// somehow refactor it. The main issue here is that it needs access to push constants and storage buffers.

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

LightSample sampleLightSources(inout uint rngState, mat4x3 objectToWorld) {
    if (pc.lightSourceTriangleCount == 0) {
        return LightSample(vec3(0.0), vec3(0.0));
    }

    float u1 = randomFloat(rngState);
    float u2 = randomFloat(rngState);

    uint i = min(uint(u1 * pc.lightSourceTriangleCount), pc.lightSourceTriangleCount - 1);

    uint triangleIndex;
    if (u2 < lightSourceAliasTableData.values[i].probability) {
        triangleIndex = i;
    } else {
        triangleIndex = lightSourceAliasTableData.values[i].alias;
    }

    uint meshId = lightSourceAliasTableData.values[triangleIndex].meshId;
    uint primitiveId = lightSourceAliasTableData.values[triangleIndex].primitiveId;

    MeshTriangle light = unpackInstanceVertex(meshId, primitiveId);
    light.v0.p = vec3(objectToWorld * vec4(light.v0.p, 1.0));
    light.v1.p = vec3(objectToWorld * vec4(light.v1.p, 1.0));
    light.v2.p = vec3(objectToWorld * vec4(light.v2.p, 1.0));

    vec3 position = sampleTriangleUniform(rngState, light.v0.p, light.v1.p, light.v2.p);
    vec3 normal   = normalize(cross(light.v1.p - light.v0.p, light.v2.p - light.v0.p));

    return LightSample(position, normal);
}

float getPdfValue(uint pdfType, vec3 direction, HitRecord rec, LightSample lightSample) {
    float cosTheta;
    switch (pdfType) {
        case SPHERE_PDF:
            return 1.0 / (4.0 * PI);
        case COSINE_PDF:
            cosTheta = dot(normalize(direction), rec.normal);
            return max(0.0, cosTheta / PI);
        case LIGHT_PDF:
            float distanceSquared = dot(direction, direction);
            cosTheta = abs(dot(lightSample.normal, -normalize(direction)));
            if (cosTheta <= 0.0) {
                return 0.0;
            }
            return (distanceSquared / cosTheta) * (1.0 / pc.lightSourceTotalArea);
        default:
            0.0;
    }
}

vec3 genScatterDirection(inout uint rngState, uint pdfType, HitRecord rec, mat4x3 objectToWorld, LightSample lightSample) {
    switch (pdfType) {
        case SPHERE_PDF:
            return randomUnitVec3(rngState);
        case COSINE_PDF:
            ONB onb = createOrthonormalBases(rec.normal);
            return onbTransform(onb, randomVec3CosineDirection(rngState));
        case LIGHT_PDF:
            return lightSample.position - rec.meshVertex.p;
        default:
            return vec3(0.0);
    }
}

uint chooseMixturePdf(inout uint rngState, uint matPdfType) {
    // No lights, fallback to material PDF.
    if (pc.lightSourceTriangleCount == 0 || pc.lightSourceTotalArea <= 0.0) {
        return matPdfType;
    }

    // 50-50 mixture.
    float r = randomFloat(rngState);
    return (r < 0.5) ? LIGHT_PDF : matPdfType;
}

ScatterRecord lambertianMaterialScatter(inout uint rngState, uint materialIndex, HitRecord rec) {
    ScatterRecord srec = initScatterRecord();

    if (materialIndex >= 0 && materialIndex < pc.lambertianMaterialCount) {
        LambertianMaterial material = lambertianMaterial.values[materialIndex];
        vec3 albedo = getMaterialPropertyValue(material.albedo, rec.meshVertex);

        srec.attenuation = albedo;
        srec.isScattered = true;
        srec.skipPdf     = false;
        srec.matPdfType  = COSINE_PDF;
    }

    return srec;
}

ScatterRecord metalMaterialScatter(inout uint rngState, uint materialIndex, HitRecord rec, vec3 worldRayDirection, float time) {
    ScatterRecord srec = initScatterRecord();

    if (materialIndex >= 0 && materialIndex < pc.metalMaterialCount) {
        MetalMaterial material = metalMaterial.values[materialIndex];
        vec3 albedo = getMaterialPropertyValue(material.albedo, rec.meshVertex);
        vec3 fuzz = getMaterialPropertyValue(material.fuzz, rec.meshVertex);

        vec3 reflectedDirection = reflect(worldRayDirection, rec.normal);

        srec.attenuation          = albedo;
        srec.isScattered          = dot(reflectedDirection, rec.normal) > 0;
        srec.matPdfType           = NO_PDF;
        srec.skipPdf              = true;
        srec.skipPdfRay.origin    = rec.meshVertex.p;
        srec.skipPdfRay.direction = normalize(reflectedDirection) + (fuzz * randomUnitVec3(rngState));
        srec.skipPdfRay.time      = time;
    }

    return srec;
}

ScatterRecord dielectricMaterialScatter(inout uint rngState, uint materialIndex, HitRecord rec, vec3 worldRayDirection, float time) {
    ScatterRecord srec = initScatterRecord();

    if (materialIndex >= 0 && materialIndex < pc.dielectricMaterialCount) {
        DielectricMaterial material = dielectricMaterial.values[materialIndex];
        float refractionIndex = material.refractionIndex;

        vec3 attenuation = vec3(1.0);

        float ri = rec.isFrontFace ? (1.0 / refractionIndex) : refractionIndex;

        vec3 unitDirection = normalize(worldRayDirection);

        float cosTheta = min(dot(-unitDirection, rec.normal), 1.0);
        float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

        bool cannotRefract = ri * sinTheta > 1.0; 
        cannotRefract = cannotRefract || schlickReflectance(cosTheta, ri) > randomFloat(rngState);

        vec3 refractedDirection = cannotRefract
            ? reflect(unitDirection, rec.normal) // Total internal reflection.
            : refract(unitDirection, rec.normal, ri);

        srec.attenuation          = attenuation;
        srec.isScattered          = true;
        srec.matPdfType           = NO_PDF;
        srec.skipPdf              = true;
        srec.skipPdfRay.origin    = rec.meshVertex.p;
        srec.skipPdfRay.direction = refractedDirection;
        srec.skipPdfRay.time      = time;
    }

    return srec;
}

EmissionRecord diffuseLightMaterialEmission(inout uint rngState, uint materialIndex, HitRecord rec) {
    EmissionRecord erec =  initEmissionRecord();

    if (materialIndex >= 0 && materialIndex < pc.diffuseLightMaterialCount) {
        DiffuseLightMaterial material = diffuseLightMaterial.values[materialIndex];
        if (rec.isFrontFace) {
            erec.emissionColour = getMaterialPropertyValue(material.emit, rec.meshVertex);
        }
    }

    return erec;
}

ScatterRecord calculateScatter(inout uint rngState, Material material, HitRecord rec, vec3 worldRayDirection, float time) {
    switch (material.type) {
        case MAT_TYPE_LAMBERTIAN:
            return lambertianMaterialScatter(rngState, material.index, rec);

        case MAT_TYPE_METAL:
            return metalMaterialScatter(rngState, material.index, rec, worldRayDirection, time);

        case MAT_TYPE_DIELECTRIC:
            return dielectricMaterialScatter(rngState, material.index, rec, worldRayDirection, time);

        default:
            // Materials that don't support scattering.
            return initScatterRecord();
    }
}

EmissionRecord calculateEmission(inout uint rngState, Material material, HitRecord rec) {
    switch (material.type) {
        case MAT_TYPE_DIFFUSE_LIGHT:
            return diffuseLightMaterialEmission(rngState, material.index, rec);

        default:
            // Non-emissive materials.
            return initEmissionRecord();
    }
}

void main() {
    rp.isMissed = false;

    MeshTriangle hitTriangle = unpackInstanceVertex(gl_InstanceCustomIndexEXT, gl_PrimitiveID);

    rp.rec = getIntersection(hitTriangle, hitAttribs, gl_ObjectToWorldEXT, gl_WorldToObjectEXT, gl_WorldRayDirectionEXT);

    Material material = unpackInstanceMaterial(gl_InstanceCustomIndexEXT);

    rp.erec = calculateEmission(rp.rngState, material, rp.rec);

    rp.srec = calculateScatter(rp.rngState, material, rp.rec, gl_WorldRayDirectionEXT, rp.time);
    if (!rp.srec.isScattered || rp.srec.skipPdf) {
        return;
    }

    // Get a the light source sample.
    LightSample lightSample = sampleLightSources(rp.rngState, gl_ObjectToWorldEXT);

    // Choose between material and light PDF with a 50-50 chance.
    uint chosenPdfType = chooseMixturePdf(rp.rngState, rp.srec.matPdfType);
    rp.srec.scatterDirection = genScatterDirection(rp.rngState, chosenPdfType, rp.rec, gl_ObjectToWorldEXT, lightSample);

    // Use material PDFs.
    float scatteringPdf = getPdfValue(rp.srec.matPdfType, rp.srec.scatterDirection, rp.rec, lightSample);
    float pdfMat        = scatteringPdf;
    float pdfValue      = pdfMat;

    // See if we want to use a Mixture PDF.
    if (pc.lightSourceTriangleCount > 0 && pc.lightSourceTotalArea > 0.0) {
        float pdfLight = getPdfValue(LIGHT_PDF, rp.srec.scatterDirection, rp.rec, lightSample);
        pdfValue = 0.5 * pdfLight + 0.5 * pdfMat;
    }

    rp.srec.attenuation *= scatteringPdf / pdfValue;
}

