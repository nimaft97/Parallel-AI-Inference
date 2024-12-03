cmake_minimum_required(VERSION 3.4)

# Use a variable for the project name to avoid repetition
set(PROJECT_NAME AICompiler)

project(${PROJECT_NAME} LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Define the common source files - keep updated
set(COMMON_SOURCES
    src/core/nn/common.cpp
    src/core/nn/model/Model.cpp
    src/core/nn/activation/Activation.cpp
    src/core/nn/layer/Layer.cpp
    src/core/nn/layer/Dense.cpp
    src/core/nn/layer/Conv2D.cpp
    src/core/nn/layer/Activation.cpp
    src/bindings/bindings.cpp
)

# Find OpenCL (cross-platform)
find_package(OpenCL REQUIRED)

# Optionally find CUDA (for future migration)
# Optionally find CUDA (for future migration)
find_package(CUDA)

# Include directories
include_directories(src/core src/gpu src/bindings)

# Add core C++ files
add_executable(model_inference src/main.cpp ${COMMON_SOURCES})

# Link OpenCL to the project
target_link_libraries(model_inference OpenCL::OpenCL)
target_compile_definitions(model_inference PRIVATE CL_TARGET_OPENCL_VERSION=100)

# Add GPU kernels
add_library(opencl_kernels src/gpu/inference_opencl.cpp)
target_link_libraries(opencl_kernels OpenCL::OpenCL)

# For PyBind11 Python bindings (once we add it)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
add_library(pybindings MODULE src/bindings/bindings.cpp)
target_link_libraries(pybindings ${PYTHON_LIBRARIES})

# Enable CUDA support in the future
if(CUDA_FOUND)
    message(STATUS "CUDA found! Enabling CUDA support.")
    enable_language(CUDA)
    add_executable(cuda_inference src/gpu/inference_cuda.cu)
    target_link_libraries(cuda_inference ${CUDA_LIBRARIES})
endif()

# Include Catch2 for testing
include(FetchContent)
FetchContent_Declare(
    catch
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1  # You can choose the latest tag
)

FetchContent_MakeAvailable(catch)

# Add unit tests
add_executable(tests_app tests/test_tensor.cpp ${COMMON_SOURCES})

target_link_libraries(tests_app PRIVATE Catch2::Catch2)
target_link_libraries(tests_app PRIVATE Catch2::Catch2WithMain)

# Link OpenCL to the project
target_link_libraries(tests_app PRIVATE OpenCL::OpenCL)
target_compile_definitions(tests_app PRIVATE CL_TARGET_OPENCL_VERSION=100)

# Add a custom target for running tests
add_custom_target(run_tests
    COMMAND tests_app
    DEPENDS tests_app
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
