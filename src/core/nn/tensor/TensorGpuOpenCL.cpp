#include "TensorGpuOpenCL.h"

#include <stdexcept>
#include <iostream>
#include <fstream>
#include <streambuf>

TensorGpuOpenCL::TensorGpuOpenCL(const dimVec& dimensions): Tensor(dimensions)
{
    // this class only supports 2D tensors (Matrix)
    if (dimensions.size() != 2)
    {
        throw std::invalid_argument("TensorGpuOpenCL only supports Matrix");
    }

    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &m_cl_device, nullptr);
    m_cl_context = clCreateContext(nullptr, 1, &m_cl_device, nullptr, nullptr, nullptr);
    m_cl_queue = clCreateCommandQueue(m_cl_context, m_cl_device, 0, nullptr);
}

TensorGpuOpenCL::~TensorGpuOpenCL()
{
    // release OpenCL 
    clReleaseCommandQueue(m_cl_queue);
    clReleaseContext(m_cl_context);
}


std::string TensorGpuOpenCL::readFile(const std::string& file_name) const {
    std::ifstream file(file_name);
    if (!file.is_open())
    {
        throw std::runtime_error("Couldn't open kernels file");
    }
    // Read the entire file into a string
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

cl_kernel TensorGpuOpenCL::loadKernel(const std::string& file_path, const std::string& kernel_name) const
{
    const std::string kernel_source = readFile(file_path).c_str();
    const char* kernel_source_str = readFile(file_path).c_str();
    const auto kernel_source_size = kernel_source.size();

    cl_int cl_ret_code;
    cl_program program = clCreateProgramWithSource(m_cl_context, 1, &kernel_source_str, &kernel_source_size, nullptr);
    cl_ret_code = clBuildProgram(program, 1, &m_cl_device, nullptr, nullptr, nullptr);
    if (cl_ret_code != CL_SUCCESS)
    {
        throw std::runtime_error("Error building OpenCL kernel from file");
    }

    cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &cl_ret_code);
    if (cl_ret_code != CL_SUCCESS)
    {
        throw std::runtime_error("Error creating kernel");
    }

    return kernel;
}