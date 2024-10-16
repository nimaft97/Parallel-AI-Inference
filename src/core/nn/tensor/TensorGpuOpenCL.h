#ifndef TENSOR_GPU_OPENCL_H
#define TENSOR_GPU_OPENCL_H

#include "Tensor.h"

#include <CL/cl.hpp>
#include <string>

class TensorGpuOpenCL : public Tensor
{
public:
    TensorGpuOpenCL(const dimVec& dimensions);
    ~TensorGpuOpenCL();
    
    TensorGpuOpenCL multiplyOnGpu(const TensorGpuOpenCL& other) const;

protected:
    cl_kernel loadKernel(const std::string& file_path, const std::string& kernel_name) const;
    std::string readFile(const std::string& file_path) const;

private:
    cl_context m_cl_context;
    cl_command_queue m_cl_queue;
    cl_device_id m_cl_device;
};

#endif