#ifndef CNCRT_SHMEM_H
#define CNCRT_SHMEM_H

#include "helper_cuda.h"
#include <atomic>
#include <iostream>
#include <mutex>
#include <thread>
#include <tuple>
#include <vector>

class SharedMemory {
public:
  char *m_memory_block;
  int m_last_byte;

  __device__ SharedMemory(char *ptr) : m_memory_block(ptr), m_last_byte(0) {}

  template <typename T> __device__ void get_allocation(T **ptr, int elements) {
    *ptr = (T *)(&this->m_memory_block[m_last_byte]);
    this->m_last_byte += elements * sizeof(T);
  }
};

class DeviceMemory {
public:
  std::vector<std::tuple<void *, int>> m_allocated;
  std::mutex m_allocation_mtx;
  std::atomic<uint32_t> m_total_devices;

  DeviceMemory() : m_total_devices(1) {}

  __host__ void set_device(int device) {
    if (device > m_total_devices)
      m_total_devices = device + 1;
  }

  template <typename T>
  __host__ void get_allocation(T **ptr, int elements, int device) {
    T *res;
    cudaMalloc((void **)&res, sizeof(T) * elements);
    *ptr = res;
    std::lock_guard<std::mutex> lock(m_allocation_mtx);
    m_allocated.push_back(std::make_tuple(res, device));
  }

  template <typename T>
  __host__ void get_allocation_and_copy_async(T **ptr, T *src, int elements,
                                              int device) {
    T *res;
    cudaMalloc((void **)&res, sizeof(T) * elements);
    cudaMemcpyAsync(res, src, sizeof(T) * elements, cudaMemcpyHostToDevice);
    *ptr = res;
    std::lock_guard<std::mutex> lock(m_allocation_mtx);
    m_allocated.push_back(std::make_tuple(res, device));
  }

  template <typename T>
  __host__ void get_allocation_and_copy_async(T **ptr, T *src, int allocation,
                                              int elements, int device) {
    T *res;
    cudaMalloc((void **)&res, sizeof(T) * allocation);
    cudaMemcpyAsync(res, src, sizeof(T) * elements, cudaMemcpyHostToDevice);
    *ptr = res;
    std::lock_guard<std::mutex> lock(m_allocation_mtx);
    m_allocated.push_back(std::make_tuple(res, device));
  }

  void free_all_from_device(int device) {
    cudaSetDevice(device);
    for (auto elem : m_allocated) {
      auto dev = std::get<1>(elem);
      if (dev == device) {
        auto mem = std::get<0>(elem);
        checkCudaErrors(cudaFree(mem));
      }
    }
  }

  __host__ ~DeviceMemory() {
    for (auto elem : m_allocated) {
      auto dev = std::get<1>(elem);
      auto mem = std::get<0>(elem);
      cudaSetDevice(dev);
      checkCudaErrors(cudaFree(mem));
    }
  }
};

#endif // CNCRT_SHMEM_H
