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
    TensorOpenCL(const cl_program& program, const cl_command_queue& queue, const cl_context& context);
    virtual ~TensorOpenCL();

    virtual void load_to_device() override;
    virtual void load_to_host() override;
    virtual Tensor<DATA_T> add_on_device(const Tensor<DATA_T>& other) const;
    virtual Tensor<DATA_T> multiply_on_device(const Tensor<DATA_T>& other) const;

protected:
    virtual std::unique_ptr<Tensor<DATA_T>> clone() const override;

private:
    void release_device_data();

private:
    cl_mem m_device_data = nullptr;
    cl_program m_program;
    cl_command_queue m_queue;
    cl_context m_context;
    mutable cl_int m_err = CL_SUCCESS;
};

template<typename DATA_T>
TensorOpenCL<DATA_T>::TensorOpenCL(const TensorOpenCL<DATA_T>& other): Tensor<DATA_T>(other)
{
    m_program = other.m_program;
    m_queue   = other.m_queue;
    m_context = other.m_context;

    // if other has data on the GPU side
    // make a deep copy of the device data instead of pointing other.m_device_data
    if (other.get_platform() == PLATFORM::DEVICE)
    {
        m_device_data = clCreateBuffer(m_context, CL_MEM_READ_WRITE, m_size * sizeof(DATA_T), nullptr, &m_err);
        CHECK_CL_ERROR(m_err, "Couldn't allocate device buffer");

        m_err = clEnqueueCopyBuffer(m_queue, other.m_device_data, m_device_data,
                                            0, 0, m_size * sizeof(DATA_T), 0, nullptr, nullptr);
        CHECK_CL_ERROR(m_err, "Couldn't copy device buffer");
    }
}

template<typename DATA_T>
TensorOpenCL<DATA_T>::TensorOpenCL(const cl_program& program, const cl_command_queue& queue, const cl_context& context): Tensor<DATA_T>()
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
std::unique_ptr<Tensor<DATA_T>> TensorOpenCL<DATA_T>::clone() const
{
    return std::make_unique<TensorOpenCL<DATA_T>>(TensorOpenCL<DATA_T>(*this));
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
    std::cerr << "load_to_host TensorOpenCL\n";
    const auto size_in_byte = m_size * sizeof(DATA_T);
    m_err = clEnqueueReadBuffer(m_queue, m_device_data, CL_TRUE, 0, size_in_byte, m_host_data.data(), 0, NULL, NULL);
    CHECK_CL_ERROR(m_err, "Couldn't write device data back to host");

    Tensor<DATA_T>::load_to_host();
}

// this function re-allocates a gpu buffer
template<typename DATA_T>
void TensorOpenCL<DATA_T>::load_to_device()
{
    release_device_data();

    // todo: not all buffers need to be read/write
    // host data cannot be empty

    const auto size_in_byte = m_size * sizeof(DATA_T);

    m_device_data = clCreateBuffer(m_context, CL_MEM_READ_WRITE, size_in_byte, NULL, &m_err);
    CHECK_CL_ERROR(m_err, "Couldn't create device buffer");

    // transfer data from host to device
    m_err = clEnqueueWriteBuffer(m_queue, m_device_data, CL_TRUE, 0, size_in_byte, m_host_data.data(), 0, NULL, NULL);
    CHECK_CL_ERROR(m_err, "Couldn't write host data to device buffer");

    if (!m_device_data)
    {
        throw std::runtime_error("device buffer is null");
    }

    Tensor<DATA_T>::load_to_device();
}

template<typename DATA_T>
Tensor<DATA_T> TensorOpenCL<DATA_T>::add_on_device(const Tensor<DATA_T>& other) const
{
    if (!is_operation_valid(*this, other, PLATFORM::DEVICE))
    {
        throw std::invalid_argument("Not all tensors are on the same platform");
    }
    const TensorOpenCL<DATA_T>* other_ptr = dynamic_cast<const TensorOpenCL<DATA_T>*>(&other);
    TensorOpenCL<DATA_T> result = TensorOpenCL<DATA_T>(*this);

    if (!other_ptr)
    {
        throw std::invalid_argument("Wasn't able to downcast provided input");
    }

    // create kernel
    cl_kernel kernel = clCreateKernel(m_program, "matSum", &m_err);
    CHECK_CL_ERROR(m_err, "Couldn't create the matSum kernel");

    // set kernel args
    m_err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &m_device_data);
    CHECK_CL_ERROR(m_err, "Couldn't set arg 1");
    m_err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &(other_ptr->m_device_data));
    CHECK_CL_ERROR(m_err, "Couldn't set arg 2");
    m_err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &(result.m_device_data));
    CHECK_CL_ERROR(m_err, "Couldn't set arg 3");
    m_err = clSetKernelArg(kernel, 3, sizeof(m_size), &m_size);
    CHECK_CL_ERROR(m_err, "Couldn't set arg 4");

    size_t global_size = 32u;

    // enqueue the kernel for execution
    m_err = clEnqueueNDRangeKernel(m_queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    CHECK_CL_ERROR(m_err, "Couldn't launch the prefixSum kernel");

    return result;
}

template<typename DATA_T>
Tensor<DATA_T> TensorOpenCL<DATA_T>::multiply_on_device(const Tensor<DATA_T>& other) const
{
    if (!is_operation_valid(*this, other, PLATFORM::DEVICE))
    {
        throw std::invalid_argument("Not all tensors are on the same platform");
    }
    // todo: override in GPU-level derived classes
    return other;
}

#endif  // TENSOR_OPENCL_H