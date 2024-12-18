#include "nn/tensor/TensorOpenCL.h"
#include "nn/model/Model.h"
#include "nn/layer/Dense.h"
#include "nn/layer/Activation.h"

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
        // Query local memory size
        cl_ulong localMemSize;
        err = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &localMemSize, nullptr);
        CHECK_CL_ERROR(err, "Error querying device info.");
        std::cout << "Maximum local memory size per work group: " << localMemSize / 1024 << " KB" << std::endl;
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
    // ****************** Start of Diagnostic Logs ******************
    // Query the build log
    size_t log_size;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    char *log = new char[log_size];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
    // Print the build log
    std::cerr << "Build log:\n" << log << std::endl;
    delete[] log;
    // ****************** End of Diagnostic Logs ******************
    CHECK_CL_ERROR(err, "Couldn't build the program"); 

    auto input = new TensorOpenCL<float>(program, queue, context);
    input->set_host_data({1.0f, 2.0f, 3.0f});
    input->set_dims({1, 3});
    input->load_to_device();

    auto result = new TensorOpenCL<float>(program, queue, context);
    result->set_host_data({0.0f, 0.0f, 0.0f,
                           0.0f, 0.0f, 0.0f,
                           0.0f, 0.0f, 0.0f});
    result->set_dims({3, 3});
    result->load_to_device();

    // dense 1
    auto weight1 = new TensorOpenCL<float>(program, queue, context);
    weight1->set_host_data({3.0f, 2.0f, 1.0f,
                            6.0f, 5.0f, 4.0f,
                            2.0f, 3.0f, 4.0f});
    weight1->set_dims({3, 3});

    auto bias1  = new TensorOpenCL<float>(program, queue, context);
    bias1->set_host_data({5.0f, 4.0f, 3.0f});
    bias1->set_dims({1, 3});
    
    auto dense1 = new Dense();
    dense1->set_weight(weight1);
    dense1->set_bias(bias1);

    // dense 2
    auto weight2 = new TensorOpenCL<float>(program, queue, context);
    weight2->set_host_data({-3.0f, 2.0f,
                            2.0f, 1.0f,
                            -1.0f, 4.0f});
    weight2->set_dims({3, 2});

    auto bias2  = new TensorOpenCL<float>(program, queue, context);
    bias2->set_host_data({5.0f, 4.0f});
    bias2->set_dims({1, 2});
    
    auto dense2 = new Dense();
    dense2->set_weight(weight2);
    dense2->set_bias(bias2);

    auto relu1 = new Activation(ACTIVATION::RELU);
    auto argmax1 = new Activation(ACTIVATION::ARGMAX);

    auto model = Model();
    model.add_layer(dense1);
    model.add_layer(relu1);
    model.add_layer(dense2);
    model.add_layer(argmax1);
    model.to_device();
    model.execute(input, result);

    clFinish(queue);

    result->load_to_host();

    std::cout << "input: "    << input->to_string(true, true, true, true);
    std::cout << "result: "   << result->to_string(true, true, true, true);
}