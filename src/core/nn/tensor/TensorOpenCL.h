#ifndef TENSOR_OPENCL_H
#define TENSOR_OPENCL_H

#include "Tensor.h"

#include <CL/cl.h>

#define CHECK_CL_ERROR(err, msg) assert(err == CL_SUCCESS && msg)

template<typename DATA_T>
class TensorOpenCL : public Tensor<DATA_T>
{
public:
    TensorOpenCL(const TensorOpenCL& other);
    TensorOpenCL(cl_program program, cl_command_queue queue, cl_context context);
    ~TensorOpenCL();

    virtual void load_to_device() override;
    virtual void load_to_host() override;
    virtual void add_on_device(const Tensor<DATA_T>& other, Tensor<DATA_T>& result) const override;
    virtual void multiply_on_device(const Tensor<DATA_T>& other, Tensor<DATA_T>& result) const override;

private:
    void release_device_data();

private:
    cl_mem m_device_data = nullptr;
    cl_program m_program;
    cl_command_queue m_queue;
    cl_context m_context;
    cl_int m_err = CL_SUCCESS;
};

template<typename DATA_T>
TensorOpenCL<DATA_T>::TensorOpenCL(cl_program program, cl_command_queue queue, cl_context context): Tensor<DATA_T>()
{
    m_queue = queue;
    m_program = program;
    m_context = context;
}

template<typename DATA_T>
TensorOpenCL<DATA_T>::~TensorOpenCL()
{
    // release resources
    release_device_data();
}

template<typename DATA_T>
void TensorOpenCL<DATA_T>::release_device_data()
{
    if (m_device_data)
    {
        m_err = clReleaseMemObject(m_device_data);
        CHECK_CL_ERROR(m_err, "Couldn't release device buffer");
    }
}

template<typename DATA_T>
void TensorOpenCL<DATA_T>::load_to_host()
{
    // todo: override in GPU-level derived classes
    
}

// this function re-allocates a gpu buffer
template<typename DATA_T>
void TensorOpenCL<DATA_T>::load_to_device()
{
    release_device_data();

    // todo: not all buffers need to be read/write
    // host data cannot be empty
    const auto size_in_byte = m_host_data.size() * sizeof(DATA_T);
    m_device_data = clCreateBuffer(m_context, CL_MEM_READ_WRITE, size_in_byte, NULL, &m_err);
    CHECK_CL_ERROR(m_err, "Couldn't create device buffer");

    // transfer data from host to device
    m_err = clEnqueueWriteBuffer(m_queue, m_device_data, CL_TRUE, 0, size_in_byte, m_host_data.data(), 0, NULL, NULL);
    CHECK_CL_ERROR(m_err, "Couldn't write host data to device buffer");
}

template<typename DATA_T>
void TensorOpenCL<DATA_T>::add_on_device(const Tensor<DATA_T>& other, Tensor<DATA_T>& result) const
{
    // todo: override in GPU-level derived classes
}

template<typename DATA_T>
void TensorOpenCL<DATA_T>::multiply_on_device(const Tensor<DATA_T>& other, Tensor<DATA_T>& result) const
{
    // todo: override in GPU-level derived classes
}

#endif  // TENSOR_OPENCL_H