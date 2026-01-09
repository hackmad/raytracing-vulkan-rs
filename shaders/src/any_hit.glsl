#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable

#include "common.glsl"
#include "perlin.glsl"

layout(location = 0) rayPayloadInEXT RayPayload rp;
hitAttributeEXT VolumeHitAttribs hitAttribs;

layout(set = 4, binding = 0) uniform sampler imageTextureSampler;
layout(set = 4, binding = 1) uniform texture2D imageTextures[];

layout(set = 5, binding = 0, scalar) buffer ConstantColours {
    vec3 values[];
} constantColour;

layout(set = 6, binding = 4, scalar) buffer IsotropicMaterials {
    IsotropicMaterial values[];
} isotropicMaterial;

layout(set = 7, binding = 0, scalar) buffer CheckerTextures {
    CheckerTexture values[];
} checkerTexture;
layout(set = 7, binding = 1, scalar) buffer NoiseTextures {
    NoiseTexture values[];
} noiseTexture;

layout(set = 10, binding = 0, scalar) buffer Volumes {
    Volume values[];
} volumeData;
layout(set = 10, binding = 2, scalar) buffer ConstantMedia {
    ConstantMedium values[];
} constantMediumData;

// Make sure to check layout offsets for push constants in each of the shader files.
// The order of these matter so make sure they are consistent across all shaders.
layout(push_constant) uniform PushConstants {
    layout(offset = 72) uint isotropicMaterialCount;
    layout(offset = 76) uint constantMediaCount;
    layout(offset = 80) uint imageTextureCount;
    layout(offset = 84) uint constantColourCount;
    layout(offset = 88) uint checkerTextureCount;
    layout(offset = 92) uint noiseTextureCount;
} pc;

// NOTE: getBasicTextureValue() and getMaterialPropertyValue() are duplicated from closest_hit. Would be nice to
// somehow refactor it. The main issue here is that it needs access to push constants and storage buffers.

// This only handles constant colour, image and noise textures. Other textures like checker texture can reference
// these "basic" textures for their own properties.
vec3 getBasicTextureValue(MaterialPropertyValue matPropValue, vec3 p, vec2 uv) {
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
                        uv
                        ).rgb; // Ignore alpha for now.
            }
            break;

        case MAT_PROP_VALUE_TYPE_NOISE:
            if (matPropValue.index >= 0 && matPropValue.index < pc.noiseTextureCount) {
                float scale = noiseTexture.values[matPropValue.index].scale;
                colour = vec3(0.5, 0.5, 0.5) * (1.0 + sin(scale * p.z + 10 * turbulence(p, 7)));
            }
            break;
    }

    return colour;
}
vec3 getMaterialPropertyValue(MaterialPropertyValue matPropValue, vec3 p, vec2 uv) {
    vec3 colour = vec3(0.0);

    switch (matPropValue.propValueType) {
        case MAT_PROP_VALUE_TYPE_RGB:
        case MAT_PROP_VALUE_TYPE_IMAGE:
        case MAT_PROP_VALUE_TYPE_NOISE:
            colour = getBasicTextureValue(matPropValue, p, uv);
            break;

        case MAT_PROP_VALUE_TYPE_CHECKER:
            if (matPropValue.index >= 0 && matPropValue.index < pc.checkerTextureCount) {
                CheckerTexture texture = checkerTexture.values[matPropValue.index];

                float invScale = 1.0 / texture.scale;
                int xInteger = int(floor(invScale * p.x));
                int yInteger = int(floor(invScale * p.y));
                int zInteger = int(floor(invScale * p.z));

                bool isEven = (xInteger + yInteger + zInteger) % 2 == 0;

                colour = isEven 
                    ? getBasicTextureValue(texture.even, p, uv)
                    : getBasicTextureValue(texture.odd, p, uv);
            }
            break;
    }

    return colour;
}

ScatterRecord isotropicMaterialScatter(inout uint rngState, uint materialIndex, vec3 p, float time) {
    ScatterRecord srec = initScatterRecord();

    if (materialIndex >= 0 && materialIndex < pc.isotropicMaterialCount) {
        IsotropicMaterial material = isotropicMaterial.values[materialIndex];

        vec3 albedo = getMaterialPropertyValue(material.albedo, p, vec2(0)); // Ignore uv for volumetrics

        srec.attenuation          = albedo;
        srec.isScattered          = true;
        srec.matPdfType           = NO_PDF;
        srec.skipPdf              = true;
        srec.skipPdfRay.origin    = p;
        srec.skipPdfRay.direction = randomUnitVec3(rngState);
        srec.skipPdfRay.time      = time;
    }

    return srec;
}

ScatterRecord calculateScatter(inout uint rngState, Material material, vec3 p, float time) {
    switch (material.type) {
        case MAT_TYPE_ISOTROPIC:
            return isotropicMaterialScatter(rngState, material.index, p, time);

        default:
            // Materials that don't support volume scattering.
            return initScatterRecord();
    }
}

void main() {
    Volume volume = volumeData.values[hitAttribs.volumeIndex];

    float tEntry = max(hitAttribs.tEntry, RAY_EPS);
    float tExit  = min(hitAttribs.tExit, RAY_INF);

    if (tEntry < tExit) {
        vec3 rayOrigin = gl_WorldRayOriginEXT;
        vec3 rayDir    = gl_WorldRayDirectionEXT;

        float rayLength = length(rayDir);
        float distanceInside = (tExit - tEntry) * rayLength;

        bool hasMat = false;
        float hitDistance;
        Material material;
        switch (volume.mediumType) {
            case CONSTANT_MEDIUM:
                if (volume.mediumIndex >= 0 && volume.mediumIndex < pc.constantMediaCount) {
                  ConstantMedium medium = constantMediumData.values[volume.mediumIndex];
                  hitDistance = -log(randomFloat(rp.rngState)) / medium.density;

                  material = Material(medium.materialType, medium.materialIndex);
                  hasMat = true;
                }
                break;
        }

        if (hasMat && hitDistance < distanceInside) {
            // We only need t and p values for volumes. Rest are ignored for now.
            float t = tEntry + hitDistance / rayLength;
            vec3 p = rayOrigin + t * rayDir;
 
            Volume volume = volumeData.values[hitAttribs.volumeIndex];

            ScatterRecord srec = calculateScatter(rp.rngState, material, p, rp.time);
            rp.srec = srec;

            if (srec.isScattered) {
                return; // Accept reported intersection.
            }
        }

        ignoreIntersectionEXT; // Reject reported intersection.
    }
}
