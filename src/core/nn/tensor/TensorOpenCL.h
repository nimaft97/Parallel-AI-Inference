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

    virtual Tensor<DATA_T>* clone() const override;
    virtual void swap(Tensor<DATA_T>* other_ptr) override;

protected:
    virtual void add_on_device(const Tensor<DATA_T>* other, Tensor<DATA_T>* result) const override;
    virtual void multiply_on_device(const Tensor<DATA_T>* other, Tensor<DATA_T>* result) const override;
    virtual void relu_on_device(Tensor<DATA_T>* result) const override;


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
Tensor<DATA_T>* TensorOpenCL<DATA_T>::clone() const
{
    return new TensorOpenCL<DATA_T>(*this);
}

template<typename DATA_T>
void TensorOpenCL<DATA_T>::swap(Tensor<DATA_T>* other_ptr)
{
    auto other_ptr_opencl = dynamic_cast<TensorOpenCL<DATA_T>*>(other_ptr);
    if (!other_ptr_opencl)
    {
        std::cerr <<  __FILE__ << ": "<< __LINE__ << std::endl;
        throw std::runtime_error("Couldn't cast to TensorOpenCL");
    }

    this->Tensor<DATA_T>::swap(other_ptr_opencl);
    std::swap(m_device_data, other_ptr_opencl->m_device_data);
    std::swap(m_program, other_ptr_opencl->m_program);
    std::swap(m_queue, other_ptr_opencl->m_queue);
    std::swap(m_context, other_ptr_opencl->m_context);
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
    if (m_platform != PLATFORM::HOST)
    {
        const auto size_in_byte = m_size * sizeof(DATA_T);
        m_err = clEnqueueReadBuffer(m_queue, m_device_data, CL_TRUE, 0, size_in_byte, m_host_data.data(), 0, NULL, NULL);
        CHECK_CL_ERROR(m_err, "Couldn't write device data back to host");

        Tensor<DATA_T>::load_to_host();
    }
}

// this function re-allocates a gpu buffer
template<typename DATA_T>
void TensorOpenCL<DATA_T>::load_to_device()
{
    if (m_platform != PLATFORM::DEVICE)
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
            std::cerr <<  __FILE__ << ": "<< __LINE__ << std::endl;
            throw std::runtime_error("device buffer is null");
        }

        Tensor<DATA_T>::load_to_device();
    }
}

template<typename DATA_T>
void TensorOpenCL<DATA_T>::add_on_device(const Tensor<DATA_T>* other, Tensor<DATA_T>* result) const
{
    auto other_ptr = dynamic_cast<const TensorOpenCL<DATA_T>*>(other);
    auto result_ptr = dynamic_cast<TensorOpenCL<DATA_T>*>(result);

    if (!other_ptr || !result_ptr)
    {
        std::cerr <<  __FILE__ << ": "<< __LINE__ << std::endl;
        throw std::runtime_error("Couldn't cast to TensorOpenCL");
    }

    // create kernel
    cl_kernel kernel = clCreateKernel(m_program, "matSum", &m_err);
    CHECK_CL_ERROR(m_err, "Couldn't create the matSum kernel");

    // set kernel args
    m_err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &m_device_data);
    CHECK_CL_ERROR(m_err, "Couldn't set arg 1");
    m_err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &(other_ptr->m_device_data));
    CHECK_CL_ERROR(m_err, "Couldn't set arg 2");
    m_err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &(result_ptr->m_device_data));
    CHECK_CL_ERROR(m_err, "Couldn't set arg 3");
    m_err = clSetKernelArg(kernel, 3, sizeof(m_size), &m_size);
    CHECK_CL_ERROR(m_err, "Couldn't set arg 4");

    size_t global_size = 32u;

    // enqueue the kernel for execution
    m_err = clEnqueueNDRangeKernel(m_queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    CHECK_CL_ERROR(m_err, "Couldn't launch the matSum kernel");
}

template<typename DATA_T>
void TensorOpenCL<DATA_T>::multiply_on_device(const Tensor<DATA_T>* other, Tensor<DATA_T>* result) const
{
    auto other_ptr = dynamic_cast<const TensorOpenCL<DATA_T>*>(other);
    auto result_ptr = dynamic_cast<TensorOpenCL<DATA_T>*>(result);

    if (!other_ptr || !result_ptr)
    {
        std::cerr <<  __FILE__ << ": "<< __LINE__ << std::endl;
        throw std::runtime_error("Couldn't cast to TensorOpenCL");
    }

    const auto other_dims = other_ptr->get_dims();

    // create kernel
    cl_kernel kernel = clCreateKernel(m_program, "gemm", &m_err);
    CHECK_CL_ERROR(m_err, "Couldn't create the matSum kernel");

    // set kernel args
    m_err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &m_device_data);
    CHECK_CL_ERROR(m_err, "Couldn't set arg 1");
    m_err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &(other_ptr->m_device_data));
    CHECK_CL_ERROR(m_err, "Couldn't set arg 2");
    m_err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &(result_ptr->m_device_data));
    CHECK_CL_ERROR(m_err, "Couldn't set arg 3");
    m_err = clSetKernelArg(kernel, 3, sizeof(m_size), &m_dims[0]);
    CHECK_CL_ERROR(m_err, "Couldn't set arg 4");
    m_err = clSetKernelArg(kernel, 4, sizeof(m_size), &m_dims[1]);
    CHECK_CL_ERROR(m_err, "Couldn't set arg 5");
    m_err = clSetKernelArg(kernel, 5, sizeof(m_size), &other_dims[1]);
    CHECK_CL_ERROR(m_err, "Couldn't set arg 6");

    size_t global_size[] = {32u, 32u};
    size_t local_size[] = {16u, 16u};

    // enqueue the kernel for execution
    m_err = clEnqueueNDRangeKernel(m_queue, kernel, 1, NULL, global_size, local_size, 0, NULL, NULL);
    CHECK_CL_ERROR(m_err, "Couldn't launch the gemm kernel");
}

template<typename DATA_T>
void TensorOpenCL<DATA_T>::relu_on_device(Tensor<DATA_T>* result) const
{
    auto result_ptr = dynamic_cast<TensorOpenCL<DATA_T>*>(result);

    if (!result_ptr)
    {
        std::cerr <<  __FILE__ << ": "<< __LINE__ << std::endl;
        throw std::runtime_error("Couldn't cast to TensorOpenCL");
    }

    // create kernel
    cl_kernel kernel = clCreateKernel(m_program, "matRelu", &m_err);
    CHECK_CL_ERROR(m_err, "Couldn't create the matSum kernel");

    // set kernel args
    m_err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &m_device_data);
    CHECK_CL_ERROR(m_err, "Couldn't set arg 1");
    m_err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &(result_ptr->m_device_data));
    CHECK_CL_ERROR(m_err, "Couldn't set arg 2");
    m_err = clSetKernelArg(kernel, 2, sizeof(m_size), &m_size);
    CHECK_CL_ERROR(m_err, "Couldn't set arg 3");

    size_t global_size = 32u;

    // enqueue the kernel for execution
    m_err = clEnqueueNDRangeKernel(m_queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    CHECK_CL_ERROR(m_err, "Couldn't launch the matRelu kernel");
}

#endif  // TENSOR_OPENCL_H