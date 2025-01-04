# Parallel AI Inference

![AI Compiler](https://github.com/user-attachments/assets/037c0e80-06b5-4ff8-8cfc-08540439e9e4)

# About The Project

Deep learning models, particularly neural networks, are revolutionizing fields such as image recognition, natural language processing, and autonomous systems. However, as these models grow more complex, executing them efficiently on diverse hardware becomes a significant challenge. This is where AI compilers come into play.

Parallel AI Inference addresses this challenge by compiling pre-trained models and providing efficient implementations for every essential component of a deep learning model, including tensors, layers, activation functions, and the overarching model. By leveraging low-level optimizations, efficient data structures, and introducing CPU and GPU parallelism, this project bridges the gap between theoretical model design and practical, high-speed execution.

By combining these strategies, Parallel AI Inference provides a scalable solution to execute neural networks efficiently, bridging the gap between cutting-edge AI research and practical deployment.

# Getting Started

## Prerequisites
- CMake (version 3.4 or later)
- OpenCL SDK (installed and configured)
- Catch2 (for unit testing)

## Build Instructions
1. Navigate to the root directory of the project.
2. Configure the project:
`cmake -S . -B ./build -DCMAKE_BUILD_TYPE=<Release/Debug> -DCMAKE_PREFIX_PATH=<path-to-OpenCL-SDK>`
3. Build the project:
`cmake --build ./build --config Release`
4. Run the executable:
`cd ./build`
`./<Release/Debug>/model_inference.exe`
5. Example:

    ```
    cmake -S . -B .\build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="C:\Users\nimaf\OneDrive\Desktop\IE\projects\opencl-proj\OpenCL-SDK\install\"

    cmake --build .\build --config Release

    cd ./build

    .\Release\model_inference.exe
    ```