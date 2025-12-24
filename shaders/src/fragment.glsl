#version 460

#include "common.glsl"

layout(location = 0) out vec4 outColor;
layout(set = 0, binding = 0) uniform sampler2D accumTexture;

void main() {
    vec2 uv = gl_FragCoord.xy / vec2(textureSize(accumTexture, 0));
    vec3 linear = texture(accumTexture, uv).rgb;
    outColor = vec4(linearTosRGB(linear), 1.0);
}

