#version 450

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D accumTex;

vec3 linearToSRGB(vec3 c) {
    return pow(c, vec3(1.0 / 2.2));
}

void main() {
    vec2 uv = gl_FragCoord.xy / vec2(textureSize(accumTex, 0));

    vec3 linear = texture(accumTex, uv).rgb;

    outColor = vec4(linearToSRGB(linear), 1.0);
}

