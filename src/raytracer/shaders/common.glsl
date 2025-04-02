#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable

struct MeshData {
    uint64_t vertexBufferAddress;
    uint64_t indexBufferAddress;
};

struct Vertex {
    vec3 position;
    vec3 normal;
    vec2 texCoord;
};

