#include <CL/cl.h>
#include <iostream>
#include <cassert>

#define MATILESIZEIZE 128
#define TILESIZE 32

const char *kernelStr =
    "__kernel void myGEMM1(const int M, const int N, const int K,"
    "                      const __global float* A,"
    "                      const __global float* B,"
    "                      __global float* C) {"
    "    const int idx = get_global_id(0);"
    "    const int idy = get_global_id(1);"
    "    float acc = 0.0f;"
    "    for (int k=0; k<K; k++) {"
    "        acc += A[k*M + idx] * B[idy*K + k];"
    "    }"
    "    C[idy*M + idx] = acc;"
    "}";


int main(int argc, char* argv[]) {

    // Set the sizes
    int K = MATILESIZEIZE;
    int M = MATILESIZEIZE;
    int N = MATILESIZEIZE;

    // Create the matrices and initialize them with random values
    float* A = new float[M*K*sizeof(float*)];
    float* B = new float[K*N*sizeof(float*)];
    float* C = new float[M*N*sizeof(float*)];

    for (int i=0; i<M*K; i++) { A[i] = 3.6*i + i*i + 3.1; }
    for (int i=0; i<K*N; i++) { B[i] = 1.2*i + 0.01*i*i + 13.9; }
    for (int i=0; i<M*N; i++) { C[i] = 0.0; }

    // Initializing OpenCL
    cl_platform_id platform = 0;
    clGetPlatformIDs(1, &platform, NULL);
    cl_device_id device = 0;
    assert(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL) == CL_SUCCESS);
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    // Compile the kernel
    cl_program program = clCreateProgramWithSource(context, 1, &kernelStr, NULL, NULL);
    clBuildProgram(program, 0, NULL, "", NULL, NULL);

    // Check for compilation errors
    assert (clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, 0, NULL, NULL) == CL_BUILD_SUCCESS);

    // Prepare OpenCL memory objects
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY,  M*K*sizeof(float), NULL, NULL);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY,  K*N*sizeof(float), NULL, NULL);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_READ_WRITE, M*N*sizeof(float), NULL, NULL);

    // Copy matrices to the GPU
    clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, M*K*sizeof(float), A, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, K*N*sizeof(float), B, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0, M*N*sizeof(float), C, 0, NULL, NULL);

    // Configure the myGEMM kernel and set its arguments
    cl_kernel kernel = clCreateKernel(program, "myGEMM1", NULL);
    clSetKernelArg(kernel, 0, sizeof(int), (void*)&M);
    clSetKernelArg(kernel, 1, sizeof(int), (void*)&N);
    clSetKernelArg(kernel, 2, sizeof(int), (void*)&K);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&bufA);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&bufB);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&bufC);

    // Run the myGEMM kernel
    const size_t local[2] = { TILESIZE, TILESIZE };
    const size_t global[2] = { M, N };
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, NULL);

    clFinish(queue);

    // Copy the output matrix C back to the CPU memory
    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, M*N*sizeof(float), C, 0, NULL, NULL);

    // Free the OpenCL memory objects
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);

    // Clean-up OpenCL 
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseProgram(program);
    clReleaseKernel(kernel);

    // Free the host memory objects
    free(A);
    free(B);
    free(C);

    // Exit
    return 0;
}
