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

TensorGpuOpenCL TensorGpuOpenCL::multiplyOnGpu(const TensorGpuOpenCL& other) const
{
    const auto other_dims = other.getDimensions();
    if (other_dims.size() != 2 || m_dimensions.size() != 2)
    {
        throw std::invalid_argument("Only matrices can be multiplied");
    }

    if (other_dims[0] != m_dimensions[1])
    {
        throw std::invalid_argument("Matrix dimensions don't match the requested multiplication");
    }

    cl_kernel kernel = loadKernel("gpu/kernels.clh", "matMul");

    TensorGpuOpenCL result = TensorGpuOpenCL({m_dimensions[0], other_dims[1]});

    // Create OpenCL buffers for the matrices
    cl_int err = CL_SUCCESS;
    cl_mem lBuffer = clCreateBuffer(m_cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                getSize() * sizeof(float), const_cast<void*>(static_cast<const void*>(m_data.data())), &err);
    if (!err == CL_SUCCESS)
    {
        throw std::runtime_error("Error creating buffer");
    }
    cl_mem rBuffer = clCreateBuffer(m_cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                other.getSize() * sizeof(float), const_cast<void*>(static_cast<const void*>(other.m_data.data())), &err);

    if (!err == CL_SUCCESS)
    {
        throw std::runtime_error("Error creating buffer");
    }
    cl_mem resultBuffer = clCreateBuffer(m_cl_context, CL_MEM_WRITE_ONLY,
                                        result.getSize() * sizeof(float), result.m_data.data(), &err);
    if (!err == CL_SUCCESS)
    {
        throw std::runtime_error("Error creating buffer");
    }

    // set kernel args
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &lBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &rBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &resultBuffer);
    clSetKernelArg(kernel, 3, sizeof(dimType), &m_dimensions[0]);
    clSetKernelArg(kernel, 4, sizeof(dimType), &m_dimensions[1]);
    clSetKernelArg(kernel, 5, sizeof(dimType), &other_dims[1]);

    // todo optimize the numbers below
    const size_t global_size[] = {32u, 32u};
    const size_t local_size[] = {32u, 32u};

    // enqueue the kernel
    clEnqueueNDRangeKernel(m_cl_queue, kernel, 2, nullptr, global_size, local_size, 0, nullptr, nullptr);

    // wait until execution of the kernel is over
    err = clFinish(m_cl_queue);
    if (!err == CL_SUCCESS)
    {
        throw std::runtime_error("Error emptying the queue");
    }

    // read the kernel's output
    clEnqueueReadBuffer(m_cl_queue, resultBuffer, CL_TRUE, 0, result.getSize() * sizeof(float), result.m_data.data(), 0, nullptr, nullptr);

    // release resources
    clReleaseMemObject(lBuffer);
    clReleaseMemObject(rBuffer);
    clReleaseMemObject(resultBuffer);
    clReleaseKernel(kernel);

    return result;
}