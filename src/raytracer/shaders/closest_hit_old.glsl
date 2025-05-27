/*
void render() {
    MeshVertex vertex = unpackInstanceVertex(gl_InstanceID, gl_PrimitiveID);
    Material diffuseMat = unpackInstanceMaterial(gl_InstanceID, MAT_PROP_TYPE_DIFFUSE);

    // Diffuse colour.
    vec3 diffuseColour = vec3(0.0);
    switch (diffuseMat.propValueType) {
        case MAT_PROP_VALUE_TYPE_RGB:
            if (diffuseMat.index >= 0 && diffuseMat.index < pc.materialColourCount) {
                diffuseColour = material_colour.values[diffuseMat.index];
            }
            break;

        case MAT_PROP_VALUE_TYPE_TEXTURE:
            if (diffuseMat.index >= 0 && diffuseMat.index < pc.textureCount) {
                diffuseColour = texture(
                        nonuniformEXT(sampler2D(textures[diffuseMat.index], textureSampler)),
                        vertex.texCoord
                        ).rgb; // Ignore alpha for now.
            }
            break;
    }

    vec3 colour = vec3(0.0);

    for (uint lightIndex = 0; lightIndex < pc.lightCount; ++lightIndex) {
        Light light = light.values[lightIndex];
        float lightIntensity = light.intensity;
        
        float distanceToLight;  // Distance to light source.
        vec3  L;                // Vector toward the light.

        switch (light.propType) {
            case LIGHT_PROP_TYPE_POSITION: 
                L = light.positionOrDirection - vertex.position;
                distanceToLight = length(L);
                L /= distanceToLight;

                float distanceToLightSq = distanceToLight * distanceToLight;
                lightIntensity /= distanceToLightSq;

                break;

            case LIGHT_PROP_TYPE_DIRECTIONAL:
                L = normalize(light.positionOrDirection);
                distanceToLight = 10000.0; // Use a very large default for directional lights.
                break;

            default:
                break;
        }

        float dotNL = max(dot(vertex.normal, L), 0.0);

        // Ambient - TODO these should come from Rust app.
        vec3 ambient = vec3(0.2, 0.2, 0.2);
        float illum = 1;

        // Lambertian diffuse.
        vec3 diffuse = diffuseColour * dotNL;
        if (illum >= 1) {
            diffuse += ambient;
        }

        // Specular - TODO should come form Rust app.
        vec3 specular = vec3(0.0);

        float attenuation = 1.0;
        if (dotNL > 0.0) {
            float tMin = 0.001;
            float tMax = distanceToLight;

            float shadowBias = 0.01;
            float frontFacing = dot(-gl_WorldRayDirectionEXT, vertex.normal);
            vec3  origin = vertex.position + sign(frontFacing) * shadowBias * vertex.normal;

            vec3  rayDir = L;
            uint  flags  = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT;
            isShadowed = true;
            traceRayEXT(topLevelAS,  // acceleration structure
                    flags,       // rayFlags
                    0xFF,        // cullMask
                    0,           // sbtRecordOffset
                    0,           // sbtRecordStride
                    1,           // missIndex
                    origin,      // ray origin
                    tMin,        // ray min range
                    rayDir,      // ray direction
                    tMax,        // ray max range
                    1);          // payload (location = 1)
        }

        if (isShadowed) {
            attenuation = 0.3;
        }

        // TODO - refine this.
        vec3 radiance = lightIntensity * attenuation * (diffuse + specular);

        // Add contribution to the colour.
        colour += radiance;
    }

    rayPayload = linearTosRGB(colour);
}
*/
