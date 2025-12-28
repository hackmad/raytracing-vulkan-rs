#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable

#include "common.glsl"
#include "perlin.glsl"

layout(location = 0) rayPayloadEXT RayPayload rayPayload;
layout(location = 1) rayPayloadEXT bool isShadowed;

layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;

layout(set = 1, binding = 0) uniform Camera {
    mat4  viewProj;     // Camera view * projection
    mat4  viewInverse;  // Camera inverse view matrix
    mat4  projInverse;  // Camera inverse projection matrix
    float focalLength;  // Focal length of lens.
    float apertureSize; // Aperture size (diameter of lens).
} camera;

layout(set = 2, binding = 0, rgba8) uniform image2D image;

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

layout(set = 8, binding = 0) uniform SkyData {
    Sky value;
} sky;

layout(set = 9, binding = 0, scalar) buffer LightSourceAliasTable {
    LightSourceAliasTableEntry values[];
} lightSourceAliasTableData;

// If this changes, make sure to update layouts for ClosestHitPushConstants in closest_hit.glsl.
//
// NOTE: See https://nvpro-samples.github.io/vk_mini_path_tracer/extras.html#moresamples.
// It explains not exceeding 64 samples per pixel and 32 batches to avoid timeouts and long renders.
// We now do progressive rendering so 64 samples per pixel is still good and you can do higher number
// of batches. However, at some point it will be diminishing returns.
layout(push_constant) uniform RayGenPushConstants {
    layout(offset =  0) uvec2 resolution;
    layout(offset =  8) uint  samplesPerPixel;
    layout(offset = 12) uint  sampleBatch;
    layout(offset = 16) uint  maxRayDepth;
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

HitRecord getIntersection(
    MeshTriangle tri,
    vec2         hitAttribs,
    mat4x3       objectToWorld,
    mat4x3       worldToObject,
    vec3         worldRayDirection
) {
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

    const vec3 worldSpacePosition = vec3(objectToWorld * vec4(position, 1.0));
    const vec3 worldSpaceNormal = normalize(vec3(normal * worldToObject));

    bool frontFace = isFrontFace(worldRayDirection, worldSpaceNormal);

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


void lambertianMaterialScatter(inout RayPayload2 rp2, uint materialIndex, HitRecord rec) {
    if (materialIndex >= 0 && materialIndex < pc.lambertianMaterialCount) {
        LambertianMaterial material = lambertianMaterial.values[materialIndex];
        vec3 albedo = getMaterialPropertyValue(material.albedo, rec.meshVertex);

        // Calculate cosine PDF.
        ONB uvw = createOrthonormalBases(rec.normal);
        vec3 scatterDirection = onbTransform(uvw, randomVec3CosineDirection(rp2.rngState));

        float cosTheta = dot(rec.normal, normalize(scatterDirection));
        float scatteringPdf = max(0.0, cosTheta / PI);

        float pdf = dot(uvw.axis[2], scatterDirection) / PI;

        rp2.scatterColour = scatteringPdf * albedo / pdf;
        rp2.isScattered = true;
        rp2.scatteredRay.direction = scatterDirection;
        rp2.scatteredRay.origin = rec.meshVertex.p;
    }
}

void metalMaterialScatter(inout RayPayload2 rp2, uint materialIndex, HitRecord rec, vec3 worldRayDirection) {
    if (materialIndex >= 0 && materialIndex < pc.metalMaterialCount) {
        MetalMaterial material = metalMaterial.values[materialIndex];
        vec3 albedo = getMaterialPropertyValue(material.albedo, rec.meshVertex);
        vec3 fuzz = getMaterialPropertyValue(material.fuzz, rec.meshVertex);

        vec3 reflectedDirection = reflect(worldRayDirection, rec.normal);

        vec3 scatteredDirection = normalize(reflectedDirection) +
            (fuzz * randomUnitVec3(rp2.rngState));

        rp2.scatterColour = albedo;
        rp2.isScattered = (dot(scatteredDirection, rec.normal) > 0);
        rp2.scatteredRay.direction = scatteredDirection;
        rp2.scatteredRay.origin = rec.meshVertex.p;
    }
}

void dielectricMaterialScatter(inout RayPayload2 rp2, uint materialIndex, HitRecord rec, vec3 worldRayDirection) {
    if (materialIndex >= 0 && materialIndex < pc.dielectricMaterialCount) {
        DielectricMaterial material = dielectricMaterial.values[materialIndex];
        float refractionIndex = material.refractionIndex;

        vec3 attenuation = vec3(1.0);

        float ri = rec.isFrontFace ? (1.0 / refractionIndex) : refractionIndex;

        vec3 unitDirection = normalize(worldRayDirection);

        float cosTheta = min(dot(-unitDirection, rec.normal), 1.0);
        float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

        bool cannotRefract = ri * sinTheta > 1.0; 
        cannotRefract = cannotRefract || schlickReflectance(cosTheta, ri) > randomFloat(rp2.rngState);

        vec3 refractedDirection = cannotRefract
            ? reflect(unitDirection, rec.normal) // Total internal reflection.
            : refract(unitDirection, rec.normal, ri);

        rp2.scatterColour = attenuation;
        rp2.isScattered = true;
        rp2.scatteredRay.direction = refractedDirection;
        rp2.scatteredRay.origin = rec.meshVertex.p;
    }
}

void diffuseLightMaterialEmission(inout RayPayload2 rp2, uint materialIndex, HitRecord rec) {
    if (materialIndex >= 0 && materialIndex < pc.diffuseLightMaterialCount) {
        DiffuseLightMaterial material = diffuseLightMaterial.values[materialIndex];
        if (rec.isFrontFace) {
            rp2.emissionColour = getMaterialPropertyValue(material.emit, rec.meshVertex);
        }
    }
}

void calculateScatter(inout RayPayload2 rp2, MeshMaterial material, HitRecord rec, vec3 worldRayDirection) {
    switch (material.type) {
        case MAT_TYPE_LAMBERTIAN:
            lambertianMaterialScatter(rp2, material.index, rec);
            break;

        case MAT_TYPE_METAL:
            metalMaterialScatter(rp2, material.index, rec, worldRayDirection);
            break;

        case MAT_TYPE_DIELECTRIC:
            dielectricMaterialScatter(rp2, material.index, rec, worldRayDirection);
            break;

        default:
            // Materials that don't support scattering.
            rp2.isScattered = false;
            rp2.scatterColour = vec3(0.0);
            break;
    }
}

void calculateEmission(inout RayPayload2 rp2, MeshMaterial material, HitRecord rec) {
    switch (material.type) {
        case MAT_TYPE_DIFFUSE_LIGHT:
            diffuseLightMaterialEmission(rp2, material.index, rec);
            break;

        default:
            // Non-emissive materials.
            rp2.emissionColour = vec3(0.0);
            break;
    }
}

RayPayload2 calculate(inout uint rngState, RayPayload rp) {
    MeshTriangle tri = unpackInstanceVertex(rp.meshId, rp.primitiveId);
    HitRecord rec = getIntersection(tri, rp.hitAttribs, rp.objectToWorld, rp.worldToObject, rp.worldRayDirection);
    MeshMaterial material = unpackInstanceMaterial(rp.meshId);

    RayPayload2 rp2 = initRayPayload2(rngState);
    calculateScatter(rp2, material, rec, rp.worldRayDirection);
    calculateEmission(rp2, material, rec);
    return rp2;
}

vec3 rayColour(inout uint rngState, Ray ray, float tMin, float tMax, uint rayFlags) {
    uint rayDepth = 0;
    vec3 attenuation = vec3(1.0);

    while (rayDepth < pc.maxRayDepth) {
        // sbtRecordOffset, sbtRecordStride control how the hitGroupId (VkAccelerationStructureInstanceKHR::
        // instanceShaderBindingTableRecordOffset) of each instance is used to look up a hit group in the 
        // SBT's hit group array. Since we only have one hit group, both are set to 0.
        //
        // missIndex is the index, within the miss shader group array of the SBT to call if no intersection is found.
        traceRayEXT(
                topLevelAS,    // acceleration structure
                rayFlags,      // rayFlags
                0xFF,          // cullMask
                0,             // sbtRecordOffset
                0,             // sbtRecordStride
                0,             // missIndex
                ray.origin,    // ray origin
                tMin,          // ray min range
                ray.direction, // ray direction
                tMax,          // ray max range
                0);            // payload (location = 0)

        RayPayload2 rp2 = calculate(rngState, rayPayload);
        rngState = rp2.rngState;

        // Closest hit and miss shader will set rayPayload fields.
        if (!rayPayload.isMissed) {
            attenuation *= (rp2.emissionColour + rp2.scatterColour);
            if (!rp2.isScattered) break;
        } else {
            vec3 unitDirection = normalize(ray.direction);
            float a = 0.5 * (unitDirection.y + 1.0);

            switch (sky.value.skyType) {
                case SKY_TYPE_SOLID:
                    attenuation *= sky.value.solid;
                    break;
                case SKY_TYPE_VERTICAL_GRADIENT:
                    attenuation *= mix(sky.value.vTop, sky.value.vBottom, sky.value.vFactor);
                    break;
                default:
                    attenuation *= vec3(0.0);
                    break;
            }

            break;
        } 

        ray = rp2.scatteredRay;
        rayDepth++;
    }

    return attenuation;
}

Ray getRay(inout uint rngState, vec2 pixelCenter, int si, int sj, float recipSqrtSpp) {
    const vec2 offset = sampleSquareStratified(rngState, si, sj, recipSqrtSpp);
    const vec2 offsetPixelCenter = pixelCenter + offset;

    const vec2 screenUV = offsetPixelCenter / vec2(gl_LaunchSizeEXT.xy);
    vec2 d = screenUV * 2.0 - 1.0;

    vec4 origin = camera.viewInverse * vec4(0.0, 0.0, 0.0, 1.0);
    vec4 target = camera.projInverse * vec4(d.x, d.y, 1.0, 1.0);
    vec4 direction = camera.viewInverse * vec4(normalize(target.xyz), 0.0);

    if (camera.apertureSize > 0.0) {
        vec4 focalPoint = vec4(camera.focalLength * normalize(target.xyz), 1.0);

        vec2 randomLensPos = sampleUniformDiskConcentric(rngState) * camera.apertureSize / 2.0;
        origin.xy += vec2(randomLensPos.x * d.x, randomLensPos.y * d.y);

        direction = vec4((normalize((camera.viewInverse * focalPoint) - origin).xyz), 0.0);
    }

    Ray ray;
    ray.origin = origin.xyz;
    ray.direction = direction.xyz;
    return ray;
}

void main() {
    uvec2 pixel = gl_LaunchIDEXT.xy;

    uint rngState = initRNG(pc.sampleBatch, pixel, pc.resolution);

    uint rayFlags = gl_RayFlagsOpaqueEXT;
    float tMin = 0.001;
    float tMax = 10000.0;

    const vec2 pixelCenter = vec2(gl_LaunchIDEXT.xy) + vec2(0.5);

    float sqrtSpp = sqrt(float(pc.samplesPerPixel));
    float recipSqrtSpp = 1.0 / sqrtSpp;
    float spp = int(sqrtSpp) * int(sqrtSpp); // In case pc.samplesPerPixel is not a perfect square.

    vec3 summedPixelColour = vec3(0.0);
    for (int sj = 0; sj < sqrtSpp; ++sj) {
        for (int si = 0; si < sqrtSpp; ++si) {
            Ray ray = getRay(rngState, pixelCenter, si, sj, recipSqrtSpp);
            vec3 attenuation = rayColour(rngState, ray, tMin, tMax, rayFlags);
            summedPixelColour += attenuation;
        }
    }

    // Blend with the averaged image in the buffer:
    vec3 averagePixelColour = summedPixelColour / spp;
    if (pc.sampleBatch != 0) {
        vec3 imageData = imageLoad(image, ivec2(pixel)).rgb;
        averagePixelColour = (pc.sampleBatch * imageData + averagePixelColour) / (pc.sampleBatch + 1);
    }

    imageStore(image, ivec2(pixel), vec4(averagePixelColour, 1.0));
}
