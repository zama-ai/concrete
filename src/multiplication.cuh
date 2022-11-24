#ifndef CUDA_ADD_H
#define CUDA_ADD_H

#ifdef __CDT_PARSER__
#undef __CUDA_RUNTIME_H__
#include <cuda_runtime.h>
#include <helper_cuda.h>
#endif

#include "../include/helper_cuda.h"
#include "linear_algebra.h"
#include "utils/kernel_dimensions.cuh"

template <typename T>
__global__ void
cleartext_multiplication(T *output, T *lwe_input, T *cleartext_input,
                         uint32_t input_lwe_dimension, uint32_t num_entries) {

  int tid = threadIdx.x;
  if (tid < num_entries) {
    int index = blockIdx.x * blockDim.x + tid;
    int cleartext_index = index / (input_lwe_dimension + 1);
    // Here we take advantage of the wrapping behaviour of uint
    output[index] = lwe_input[index] * cleartext_input[cleartext_index];
  }
}

template <typename T>
__host__ void
host_cleartext_multiplication(void *v_stream, uint32_t gpu_index, T *output,
                              T *lwe_input, T *cleartext_input,
                              uint32_t input_lwe_dimension,
                              uint32_t input_lwe_ciphertext_count) {

  cudaSetDevice(gpu_index);
  // lwe_size includes the presence of the body
  // whereas lwe_dimension is the number of elements in the mask
  int lwe_size = input_lwe_dimension + 1;
  // Create a 1-dimensional grid of threads
  int num_blocks = 0, num_threads = 0;
  int num_entries = input_lwe_ciphertext_count * lwe_size;
  getNumBlocksAndThreads(num_entries, 512, num_blocks, num_threads);
  dim3 grid(num_blocks, 1, 1);
  dim3 thds(num_threads, 1, 1);

  auto stream = static_cast<cudaStream_t *>(v_stream);
  cleartext_multiplication<<<grid, thds, 0, *stream>>>(
      output, lwe_input, cleartext_input, input_lwe_dimension, num_entries);
  checkCudaErrors(cudaGetLastError());

  cudaStreamSynchronize(*stream);
}

#endif // CUDA_ADD_H
