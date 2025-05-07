#include <iostream>
#include <cuda.h>
#include <chrono>
#include <random>
using namespace std;

#define N 512  // Matrix size N x N

__global__ void matMulGPU(int* A, int* B, int* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int sum = 0;
        for (int k = 0; k < n; k++)
            sum += A[row * n + k] * B[k * n + col];
        C[row * n + col] = sum;
    }
}

void matMulCPU(int* A, int* B, int* C) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            int sum = 0;
            for (int k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

void initializeMatrix(int* mat) {
	random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, 1000 - 1);

    for (int i = 0; i < N * N; i++)
        mat[i] = dis(gen);
}

int main() {
    int size = N * N * sizeof(int);
    int *A, *B, *C, *C_cpu;
    int *d_A, *d_B, *d_C;

    A = (int*)malloc(size);
    B = (int*)malloc(size);
    C = (int*)malloc(size);
    C_cpu = (int*)malloc(size);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    initializeMatrix(A);
    initializeMatrix(B);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((N + 15) / 16, (N + 15) / 16);

    // GPU Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matMulGPU<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpuTime = 0;
    cudaEventElapsedTime(&gpuTime, start, stop);
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cout << "GPU Time: " << gpuTime << " ms" << endl;

    // CPU Timing
    auto cpu_start = chrono::high_resolution_clock::now();
    matMulCPU(A, B, C_cpu);
    auto cpu_end = chrono::high_resolution_clock::now();
    chrono::duration<double> cpu_time = cpu_end - cpu_start;
    cout << "CPU Time: " << cpu_time.count() * 1000 << " ms" << endl;

    // Free memory
    free(A); free(B); free(C); free(C_cpu);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}
