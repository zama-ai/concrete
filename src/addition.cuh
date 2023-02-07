#ifndef CUDA_ADD_H
#define CUDA_ADD_H

#ifdef __CDT_PARSER__
#undef __CUDA_RUNTIME_H__
#include <cuda_runtime.h>
#endif

#include "device.h"
#include "linear_algebra.h"
#include "utils/kernel_dimensions.cuh"
#include <stdio.h>

template <typename T>
__global__ void addition(T *output, T *input_1, T *input_2,
                         uint32_t num_entries) {

  int tid = threadIdx.x;
  int index = blockIdx.x * blockDim.x + tid;
  if (index < num_entries) {
    // Here we take advantage of the wrapping behaviour of uint
    output[index] = input_1[index] + input_2[index];
  }
}

template <typename T>
__global__ void plaintext_addition(T *output, T *lwe_input, T *plaintext_input,
                                   uint32_t input_lwe_dimension,
                                   uint32_t num_entries) {

  int tid = threadIdx.x;
  int plaintext_index = blockIdx.x * blockDim.x + tid;
  if (plaintext_index < num_entries) {
    int index =
        plaintext_index * (input_lwe_dimension + 1) + input_lwe_dimension;
    // Here we take advantage of the wrapping behaviour of uint
    output[index] = lwe_input[index] + plaintext_input[plaintext_index];
  }
}

template <typename T>
__host__ void host_addition(void *v_stream, uint32_t gpu_index, T *output,
                            T *input_1, T *input_2,
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
  addition<<<grid, thds, 0, *stream>>>(output, input_1, input_2, num_entries);
  check_cuda_error(cudaGetLastError());
}

template <typename T>
__host__ void host_addition_plaintext(void *v_stream, uint32_t gpu_index,
                                      T *output, T *lwe_input,
                                      T *plaintext_input,
                                      uint32_t input_lwe_dimension,
                                      uint32_t input_lwe_ciphertext_count) {

  cudaSetDevice(gpu_index);
  int num_blocks = 0, num_threads = 0;
  int num_entries = input_lwe_ciphertext_count;
  getNumBlocksAndThreads(num_entries, 512, num_blocks, num_threads);
  dim3 grid(num_blocks, 1, 1);
  dim3 thds(num_threads, 1, 1);

  auto stream = static_cast<cudaStream_t *>(v_stream);

  check_cuda_error(cudaMemcpyAsync(output, lwe_input,
                                   (input_lwe_dimension + 1) *
                                       input_lwe_ciphertext_count * sizeof(T),
                                   cudaMemcpyDeviceToDevice, *stream));
  plaintext_addition<<<grid, thds, 0, *stream>>>(
      output, lwe_input, plaintext_input, input_lwe_dimension, num_entries);
  check_cuda_error(cudaGetLastError());
}
#endif // CUDA_ADD_H
