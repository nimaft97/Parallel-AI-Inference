# Parallel AI Inference

![AI Compiler](https://github.com/user-attachments/assets/037c0e80-06b5-4ff8-8cfc-08540439e9e4)

# About The Project

Deep learning models, particularly neural networks, are revolutionizing fields such as image recognition, natural language processing, and autonomous systems. However, as these models grow more complex, executing them efficiently on diverse hardware becomes a significant challenge. This is where AI compilers come into play.

Parallel AI Inference addresses this challenge by compiling pre-trained models and providing efficient implementations for every essential component of a deep learning model, including tensors, layers, activation functions, and the overarching model. By leveraging low-level optimizations, efficient data structures, and introducing CPU and GPU parallelism, this project fills the gap between theoretical model design and high-speed inference.

By combining these strategies, Parallel AI Inference provides a scalable solution to execute neural networks efficiently, bridging the gap between cutting-edge AI research and practical deployment.

# Components

The Parallel AI Inference project is a deep learning compiler composed of the following key components:

## C++ Core

At the heart of this project lies the C++ core, which implements all the fundamental components of the deep learning compiler. C++ serves as the backbone, defining the architecture and enabling multiple levels of parallelism, from CPU-based multithreading to GPU acceleration. This core includes baseline implementations (known as HOST code) that can be used for benchmarking purposes.  

Leveraging modern, templatized, and object-oriented C++, this core ensures scalability and support for various data types while maintaining clarity with consistent terminology. The structure is designed for modularity, making it easy to extend and adapt for different machine learning architectures and datasets.  

## GPU Computation

Key C++ components such as tensors, layers, and models are architected to support both host and device-side computations seamlessly. The project uses OpenCL as its preferred GPU interface for general-purpose GPU (GPGPU) programming, ensuring cross-platform compatibility and scalability.  

OpenCL kernels provide efficient implementations of highly parallelizable algorithms, including matrix operations like GEMM (General Matrix Multiplication), as highlighted by [NVIDIA's documentation](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html). Switching between host and device computations is as simple as invoking a single function, with intelligent data management to minimize unnecessary data transfers between the host and device.  

## Maintainability

To ensure robust project management and maintainability, the project employs:  

- **CMake**: Facilitates the build process across platforms, making it easy to compile and configure the project for different environments and hardware setups.  
- **Catch2**: Provides comprehensive unit testing to validate the correctness of individual components and end-to-end functionality. The integration of Catch2 ensures code reliability, prevents regressions, and encourages test-driven development.  

Additionally, the modular design of the project promotes code reusability, enabling contributors to add new features or optimizations with minimal disruption to the existing codebase. The combination of a clear directory structure, efficient build tools, and a robust testing framework makes this project maintainable and scalable for future enhancements.  

# Getting Started

## Prerequisites
- CMake (version 3.4 or later)
- C++ (17 or higher)
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

# License
This project is licensed under the MIT License. See [LICENSE](https://github.com/nimaft97/Parallel-AI-Inference/tree/main?tab=MIT-1-ov-file) for details.


# Acknowledgments
- OpenCL for enabling cross-platform parallel computation.
- Catch2 for providing a robust unit testing framework.
- CMake for simplifying the build process.