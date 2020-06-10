//
// Created by jgleeson on 2020-05-14.
//
#ifndef CUPTI_SAMPLES_COMMONCUDA_CUH
#define CUPTI_SAMPLES_COMMONCUDA_CUH

#include <cuda.h>
#include <cuda_runtime.h>

namespace rlscope {

void ComputeVecAdd(int iters, size_t n_int32s);
void ComputeVectorAddSubtract(size_t N);
__global__ void VecAdd(const int* A, const int* B, int* C, int N);
__global__ void VecSub(const int* A, const int* B, int* C, int N);


} // namespace rlscope

#endif //CUPTI_SAMPLES_COMMONCUDA_CUH
