![](https://raw.githubusercontent.com/JunioJsv/tinyraytracer/master/out.png)

### Camera Controls

| Input             | Action                      |
| ----------------- | --------------------------- |
| **W / A / S / D** | Move camera                 |
| **Space**         | Move up                     |
| **Left Ctrl**     | Move down                   |
| **Left Alt**      | Walk                        |
| **Left Shift**    | Sprint                      |
| **Mouse Move**    | Rotate camera (Yaw / Pitch) |
| **Mouse Wheel**   | Adjust Field of View (FOV)  |


## Requirements
CMake 3.16 or higher
C++20 compatible compiler:

Linux: GCC 8+ or Clang 7+
Windows: Visual Studio 2019+ or MinGW-w64
macOS: Xcode 10+ or Clang 7+

NVIDIA CUDA

CUDA Toolkit 11.0 NVIDIA GPU with compute capability 7.5+ (RTX 20xx series or newer)

Currently configured for RTX 20xx series (compute capability 7.5)
For other GPUs, edit the CUDA_ARCHITECTURES line in CMakeLists.txt
Check your compute capability at: https://developer.nvidia.com/cuda-gpus

Raylib 6.0 is included as a submodule.

## Compilation
```sh
git clone https://github.com/JunioJsv/tinyraytracer
git submodule update --init --recursive
cd tinyraytracer
cmake -S . -B build
cmake --build build
```