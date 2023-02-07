#ifndef DEVICE_H
#define DEVICE_H

#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {
cudaStream_t *cuda_create_stream(uint32_t gpu_index);

int cuda_destroy_stream(cudaStream_t *stream, uint32_t gpu_index);

void *cuda_malloc(uint64_t size, uint32_t gpu_index);

void *cuda_malloc_async(uint64_t size, cudaStream_t *stream,
                        uint32_t gpu_index);

int cuda_check_valid_malloc(uint64_t size, uint32_t gpu_index);

int cuda_memcpy_to_cpu(void *dest, const void *src, uint64_t size,
                       uint32_t gpu_index);

int cuda_memcpy_async_to_gpu(void *dest, void *src, uint64_t size,
                             cudaStream_t *stream, uint32_t gpu_index);

int cuda_memcpy_to_gpu(void *dest, void *src, uint64_t size,
                       uint32_t gpu_index);

int cuda_memcpy_async_to_cpu(void *dest, const void *src, uint64_t size,
                             cudaStream_t *stream, uint32_t gpu_index);
int cuda_get_number_of_gpus();

int cuda_synchronize_device(uint32_t gpu_index);

int cuda_drop(void *ptr, uint32_t gpu_index);

int cuda_drop_async(void *ptr, cudaStream_t *stream, uint32_t gpu_index);

int cuda_get_max_shared_memory(uint32_t gpu_index);

int cuda_synchronize_stream(void *v_stream);

#define check_cuda_error(ans)                                                  \
  { cuda_error((ans), __FILE__, __LINE__); }
inline void cuda_error(cudaError_t code, const char *file, int line,
                       bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "Cuda error: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}
}
#endif