#include "nn/tensor/TensorOpenCL.h"
#include "nn/model/Model.h"
#include "nn/layer/Dense.h"

#include <CL/cl.h>
#include <iostream>
#include <cassert>

int main(int argc, char** argv)
{
    std::cout << "Welcome to the Parallel AI Inference project" << std::endl;

    // initialize OpenCL
    cl_int err = CL_SUCCESS;
    cl_platform_id platform_id;
    cl_device_id device;
    cl_context context;
    cl_program program;
    cl_command_queue queue;

    err = clGetPlatformIDs(1, &platform_id, NULL); 
    CHECK_CL_ERROR(err, "Couldn't get platform");

    // set the device
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err == CL_SUCCESS)
    {
        // at least one OpenCL capable GPU exists
        std::cout << "GPU found" << std::endl;
    }
    else
    {
        // default to CPU
        clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
        std::cout << "No GPU found, switched back to CPU" << std::endl;
    }
    
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_CL_ERROR(err, "Couldn't create the context");
    queue = clCreateCommandQueue(context, device, NULL, &err);
    CHECK_CL_ERROR(err, "Couldn't create the queue");

    // create a program from kernel source code
    const auto file_name = "../src/gpu/kernels.clh";
    const auto kernel_source_string = read_file(file_name);
    const char* kernel_source = kernel_source_string.c_str();
    program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    CHECK_CL_ERROR(err, "Couldn't create the program");
    // build the program
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    CHECK_CL_ERROR(err, "Couldn't build the program");

    auto t_opencl = TensorOpenCL<float>(program, queue, context);
    t_opencl.set_host_data({1.0f, 2.0f, 3.0f});
    t_opencl.set_dims({1, 3});
    t_opencl.load_to_device();
    // t_opencl.load_to_host();
    std::cout << "t_opencl: "    << t_opencl.to_string(true, true, true, true);

    auto t2_opencl = TensorOpenCL<float>(program, queue, context);
    t2_opencl.set_host_data({1.0f, 2.0f, 3.0f});
    t2_opencl.set_dims({1, 3});
    t2_opencl.load_to_device();
    // t2_opencl.load_to_host();
    std::cout << "t2_opencl: "    << t2_opencl.to_string(true, true, true, true);
    auto t3_opencl = t_opencl.add_on_device(t2_opencl);
    t3_opencl.load_to_host();
    clFinish(queue);
    std::cout << "t3_opencl: "    << t3_opencl.to_string(true, true, true, true);

    

    // auto input = Tensor<float>();
    // input.set_host_data({1.0f, 2.0f, 3.0f});
    // input.set_dims({1, 3});

    // auto weight = Tensor<float>();
    // weight.set_host_data({3.0f, 2.0f, 1.0f});
    // weight.set_dims({3, 1});

    // auto bias  = Tensor<float>();
    // bias.set_host_data({5.0f});
    // bias.set_dims({1, 1});
    
    // auto dense = Dense();
    // dense.set_weight(weight);
    // dense.set_bias(bias);

    // auto model = Model();
    // model.add_layer(&dense);

    // auto result = model.execute(input);
    

    // std::cout << "input: "    << input.to_string(true, true, true, true);
    // std::cout << "weight: "   << weight.to_string(true, true, true, true);
    // std::cout << "bias: "     << bias.to_string(true, true, true, true);
    // std::cout << "result: "   << result.to_string(true, true, true, true);
}