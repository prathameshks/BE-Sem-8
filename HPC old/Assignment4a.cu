#include <stdio.h>
#include <cuda.h>
#include <chrono>

const int iNumberOfArrayElements = 11444777;
float *hostInput1, *hostInput2, *hostOutput, *deviceInput1, *deviceInput2, *deviceOutput;

// CUDA kernel for vector addition
__global__ void vecAddGPU(float *in1, float *in2, float *out, int len)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len)
        out[i] = in1[i] + in2[i];
}

// Fill array with random float values
void fillFloatArrayWithRandomNumbers(float *arr, int len)
{
    for (int i = 0; i < len; i++)
        arr[i] = (float)rand() / RAND_MAX;
}

// CPU vector addition
void vecAddCPU(const float *arr1, const float *arr2, float *out, int len)
{
    for (int i = 0; i < len; i++)
        out[i] = arr1[i] + arr2[i];
}

void cleanup()
{
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);
    free(hostInput1);
    free(hostInput2);
    free(hostOutput);
}

int main()
{
    int size = iNumberOfArrayElements * sizeof(float);
    cudaError_t result;

    // Allocate memory on host and device
    hostInput1 = (float *)malloc(size);
    hostInput2 = (float *)malloc(size);
    hostOutput = (float *)malloc(size);

    result = cudaMalloc((void **)&deviceInput1, size);
    result = cudaMalloc((void **)&deviceInput2, size);
    result = cudaMalloc((void **)&deviceOutput, size);

    // Fill input arrays with random values
    fillFloatArrayWithRandomNumbers(hostInput1, iNumberOfArrayElements);
    fillFloatArrayWithRandomNumbers(hostInput2, iNumberOfArrayElements);

    // Copy input data from host to device
    cudaMemcpy(deviceInput1, hostInput1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2, size, cudaMemcpyHostToDevice);

    // Kernel configuration
    dim3 dimBlock(256);
    dim3 dimGrid((iNumberOfArrayElements + dimBlock.x - 1) / dimBlock.x);

    // Measure GPU execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    vecAddGPU<<<dimGrid, dimBlock>>>(deviceInput1, deviceInput2, deviceOutput, iNumberOfArrayElements);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpuTime;
    cudaEventElapsedTime(&gpuTime, start, stop);

    cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);
    printf("Time taken for Vector Addition on GPU = %.6f ms\n", gpuTime);

    for(int i = 0; i < 10; i++)
        printf("Output[%d] = %f\n", i, hostOutput[i]); // Print first 10 results

    // Measure CPU execution time
    auto start_cpu = std::chrono::high_resolution_clock::now();
    vecAddCPU(hostInput1, hostInput2, hostOutput, iNumberOfArrayElements);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpuTime = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();
    printf("Time taken for Vector Addition on CPU = %.6f ms\n", cpuTime);

    cleanup();
    return 0;
}
