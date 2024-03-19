#include "device.h"
#include <cstdint>
#include <cuda_runtime.h>

/// Unsafe function to create a CUDA stream, must check first that GPU exists
cudaStream_t *cuda_create_stream(uint32_t gpu_index) {
  cudaSetDevice(gpu_index);
  cudaStream_t *stream = new cudaStream_t;
  cudaStreamCreate(stream);
  return stream;
}

/// Unsafe function to destroy CUDA stream, must check first the GPU exists
int cuda_destroy_stream(cudaStream_t *stream, uint32_t gpu_index) {
  cudaSetDevice(gpu_index);
  cudaStreamDestroy(*stream);
  return 0;
}

/// Unsafe function that will try to allocate even if gpu_index is invalid
/// or if there's not enough memory. A safe wrapper around it must call
/// cuda_check_valid_malloc() first
void *cuda_malloc(uint64_t size, uint32_t gpu_index) {
  cudaSetDevice(gpu_index);
  void *ptr;
  cudaMalloc((void **)&ptr, size);
  check_cuda_error(cudaGetLastError());

  return ptr;
}

/// Allocates a size-byte array at the device memory. Tries to do it
/// asynchronously.
void *cuda_malloc_async(uint64_t size, cudaStream_t *stream,
                        uint32_t gpu_index) {
  cudaSetDevice(gpu_index);
  void *ptr;

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11020)
  int support_async_alloc;
  check_cuda_error(cudaDeviceGetAttribute(
      &support_async_alloc, cudaDevAttrMemoryPoolsSupported, gpu_index));

  if (support_async_alloc) {
    check_cuda_error(cudaMallocAsync((void **)&ptr, size, *stream));
  } else {
    check_cuda_error(cudaMalloc((void **)&ptr, size));
  }
#else
  check_cuda_error(cudaMalloc((void **)&ptr, size));
#endif
  return ptr;
}

/// Checks that allocation is valid
/// 0: valid
/// -1: invalid, not enough memory in device
/// -2: invalid, gpu index doesn't exist
int cuda_check_valid_malloc(uint64_t size, uint32_t gpu_index) {

  if (gpu_index >= cuda_get_number_of_gpus()) {
    // error code: invalid gpu_index
    return -2;
  }
  cudaSetDevice(gpu_index);
  size_t total_mem, free_mem;
  cudaMemGetInfo(&free_mem, &total_mem);
  if (size > free_mem) {
    // error code: not enough memory
    return -1;
  }
  return 0;
}

/// Returns
///  -> 0 if Cooperative Groups is not supported.
///  -> 1 otherwise
int cuda_check_support_cooperative_groups() {
  int cooperative_groups_supported = 0;
  cudaDeviceGetAttribute(&cooperative_groups_supported,
                         cudaDevAttrCooperativeLaunch, 0);

  return cooperative_groups_supported > 0;
}

/// Tries to copy memory to the GPU asynchronously
/// 0: success
/// -1: error, invalid device pointer
/// -2: error, gpu index doesn't exist
/// -3: error, zero copy size
int cuda_memcpy_async_to_gpu(void *dest, void *src, uint64_t size,
                             cudaStream_t *stream, uint32_t gpu_index) {
  if (size == 0) {
    // error code: zero copy size
    return -3;
  }

  if (gpu_index >= cuda_get_number_of_gpus()) {
    // error code: invalid gpu_index
    return -2;
  }
  cudaPointerAttributes attr;
  cudaPointerGetAttributes(&attr, dest);
  if (attr.device != gpu_index && attr.type != cudaMemoryTypeDevice) {
    // error code: invalid device pointer
    return -1;
  }

  cudaSetDevice(gpu_index);
  check_cuda_error(
      cudaMemcpyAsync(dest, src, size, cudaMemcpyHostToDevice, *stream));
  return 0;
}

/// Synchronizes device
/// 0: success
/// -2: error, gpu index doesn't exist
int cuda_synchronize_device(uint32_t gpu_index) {
  if (gpu_index >= cuda_get_number_of_gpus()) {
    // error code: invalid gpu_index
    return -2;
  }
  cudaSetDevice(gpu_index);
  cudaDeviceSynchronize();
  return 0;
}

int cuda_memset_async(void *dest, uint64_t val, uint64_t size,
                      cudaStream_t *stream, uint32_t gpu_index) {
  if (size == 0) {
    // error code: zero copy size
    return -3;
  }

  if (gpu_index >= cuda_get_number_of_gpus()) {
    // error code: invalid gpu_index
    return -2;
  }
  cudaPointerAttributes attr;
  cudaPointerGetAttributes(&attr, dest);
  if (attr.device != gpu_index && attr.type != cudaMemoryTypeDevice) {
    // error code: invalid device pointer
    return -1;
  }
  cudaSetDevice(gpu_index);
  cudaMemsetAsync(dest, val, size, *stream);
  return 0;
}

/// Tries to copy memory to the GPU asynchronously
/// 0: success
/// -1: error, invalid device pointer
/// -2: error, gpu index doesn't exist
/// -3: error, zero copy size
int cuda_memcpy_async_to_cpu(void *dest, const void *src, uint64_t size,
                             cudaStream_t *stream, uint32_t gpu_index) {
  if (size == 0) {
    // error code: zero copy size
    return -3;
  }

  if (gpu_index >= cuda_get_number_of_gpus()) {
    // error code: invalid gpu_index
    return -2;
  }
  cudaPointerAttributes attr;
  cudaPointerGetAttributes(&attr, src);
  if (attr.device != gpu_index && attr.type != cudaMemoryTypeDevice) {
    // error code: invalid device pointer
    return -1;
  }

  cudaSetDevice(gpu_index);
  check_cuda_error(
      cudaMemcpyAsync(dest, src, size, cudaMemcpyDeviceToHost, *stream));
  return 0;
}

/// Return number of GPUs available
int cuda_get_number_of_gpus() {
  int num_gpus;
  cudaGetDeviceCount(&num_gpus);
  return num_gpus;
}

/// Drop a cuda array
int cuda_drop(void *ptr, uint32_t gpu_index) {
  if (gpu_index >= cuda_get_number_of_gpus()) {
    // error code: invalid gpu_index
    return -2;
  }
  cudaSetDevice(gpu_index);
  check_cuda_error(cudaFree(ptr));
  return 0;
}

/// Drop a cuda array. Tries to do it asynchronously
int cuda_drop_async(void *ptr, cudaStream_t *stream, uint32_t gpu_index) {

  cudaSetDevice(gpu_index);
#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11020)
  int support_async_alloc;
  check_cuda_error(cudaDeviceGetAttribute(
      &support_async_alloc, cudaDevAttrMemoryPoolsSupported, gpu_index));

  if (support_async_alloc) {
    check_cuda_error(cudaFreeAsync(ptr, *stream));
  } else {
    check_cuda_error(cudaFree(ptr));
  }
#else
  check_cuda_error(cudaFree(ptr));
#endif
  return 0;
}

/// Get the maximum size for the shared memory
int cuda_get_max_shared_memory(uint32_t gpu_index) {
  if (gpu_index >= cuda_get_number_of_gpus()) {
    // error code: invalid gpu_index
    return -2;
  }
  cudaSetDevice(gpu_index);
  static int max_shared_memory = 0;
  if (max_shared_memory == 0) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpu_index);
    if (prop.major >= 6) {
      max_shared_memory = prop.sharedMemPerMultiprocessor;
    } else {
      max_shared_memory = prop.sharedMemPerBlock;
    }
  }
  return max_shared_memory;
}

int cuda_synchronize_stream(void *v_stream) {
  auto stream = static_cast<cudaStream_t *>(v_stream);
  cudaStreamSynchronize(*stream);
  return 0;
}
