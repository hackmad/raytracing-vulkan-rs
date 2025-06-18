# Raytracing with Vulkan

## References:

1. [vulkano raytracing example](https://github.com/vulkano-rs/vulkano/tree/0.35.X/examples/ray-tracing-auto)
2. [egui_winit_vulkano example](https://github.com/hakolao/egui_winit_vulkano/blob/master/examples/wholesome/main.rs)
3. [vk-rt](https://github.com/brunosegiu/vk-rt)
4. [Vulkan GLSL Ray Tracing Emulator](https://www.gsn-lib.org/docs/nodes/raytracing.php)
5. [Raytracing in a Weekend Series](https://raytracing.github.io/)
6. [nvpro-samples - vk_mini_path_tracer](https://nvpro-samples.github.io/vk_mini_path_tracer/index.html#antialiasingandpseudorandomnumbergeneration/pseudorandomnumbergenerationinglsl)

## Credits:

- [NASA Visible Earth](https://visibleearth.nasa.gov/images/73909/december-blue-marble-next-generation-w-topography-and-bathymetry) for the image texture.

## Suported platforms:

- Linux: Tested
  - System: Linux Mint 22.1
  - CPU: AMD 9950X
  - RAM: 64GB
  - GPU: NVIDIA RTX 4080 Super (nvidia-driver-565-open)
  - Kernel: 6.11.0-26-generic
- Windows: Untested
- MacOS: [Needs MoltenVK support for acceleration structures](https://github.com/KhronosGroup/MoltenVK/issues/1956).

## Running

Main binary:

```bash
cargo run --release
```

Generate scene file for Raytracing in a Weekend final scene:

```bash
cargo run -p tools -- gen-final-one-weekend
```
```

