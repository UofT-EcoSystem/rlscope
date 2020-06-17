//
// Created by jgleeson on 2020-05-14.
//

#include "CommonCuda.cuh"

#include <iostream>

namespace CuptiSamples {

// Device code
__global__ void VecAdd(const int* A, const int* B, int* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

// Device code
__global__ void VecSub(const int* A, const int* B, int* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] - B[i];
}

static void initVec(int *vec, int n)
{
    for (int i = 0; i < n; i++)
        vec[i] = i;
}

void ComputeVecAdd(int iters, size_t n_int32s)
{
    size_t size = n_int32s * sizeof(int);
    int threadsPerBlock = 0;
    int blocksPerGrid = 0;
    int sum;
    int *h_A, *h_B, *h_C;
    int *d_A, *d_B, *d_C;

    // Allocate input vectors h_A and h_B in host memory
    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_C = (int*)malloc(size);

    // Initialize input vectors
    initVec(h_A, n_int32s);
    initVec(h_B, n_int32s);
    memset(h_C, 0, size);

    // Allocate vectors in device memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Invoke kernel (multiple times to make sure we have time for
    // sampling)
    threadsPerBlock = 256;
    blocksPerGrid = (n_int32s + threadsPerBlock - 1) / threadsPerBlock;
    for (int i = 0; i < iters; i++) {
        VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n_int32s);
    }


    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify result
    for (size_t i = 0; i < n_int32s; ++i) {
        sum = h_A[i] + h_B[i];
        if (h_C[i] != sum) {
            std::cout << "kernel execution FAILED" << std::endl;
            exit(-1);
        }
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
}

void ComputeVectorAddSubtract(size_t N)
{
    // int N = 50000;
    size_t size = N * sizeof(int);
    int threadsPerBlock = 0;
    int blocksPerGrid = 0;
    int *h_A, *h_B, *h_C, *h_D;
    int *d_A, *d_B, *d_C, *d_D;
    int sum, diff;

    // Allocate input vectors h_A and h_B in host memory
    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_C = (int*)malloc(size);
    h_D = (int*)malloc(size);

    // Initialize input vectors
    initVec(h_A, N);
    initVec(h_B, N);
    memset(h_C, 0, size);
    memset(h_D, 0, size);

    // Allocate vectors in device memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    cudaMalloc((void**)&d_D, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Invoke kernel
    threadsPerBlock = 256;
    blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    printf("Launching kernel: blocks %d, thread/block %d\n",
           blocksPerGrid, threadsPerBlock);

    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    VecSub<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_D, N);

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_D, d_D, size, cudaMemcpyDeviceToHost);

    // Verify result
    for (size_t i = 0; i < N; ++i) {
        sum = h_A[i] + h_B[i];
        diff = h_A[i] - h_B[i];
        if (h_C[i] != sum || h_D[i] != diff) {
            fprintf(stderr, "error: result verification failed\n");
            exit(-1);
        }
    }

    if (d_A)
        cudaFree(d_A);
    if (d_B)
        cudaFree(d_B);
    if (d_C)
        cudaFree(d_C);
    if (d_D)
        cudaFree(d_D);

    // Free host memory
    if (h_A)
        free(h_A);
    if (h_B)
        free(h_B);
    if (h_C)
        free(h_C);
    if (h_D)
        free(h_D);
}

} // namespace CuptiSamples
