#include <cstdint>

extern "C" {
void *cuda_create_stream(uint32_t gpu_index);

int cuda_destroy_stream(void *v_stream, uint32_t gpu_index);

void *cuda_malloc(uint64_t size, uint32_t gpu_index);

int cuda_check_valid_malloc(uint64_t size, uint32_t gpu_index);

int cuda_memcpy_to_cpu(void *dest, const void *src, uint64_t size,
                       uint32_t gpu_index);

int cuda_memcpy_async_to_gpu(void *dest, void *src, uint64_t size,
                             void *v_stream, uint32_t gpu_index);

int cuda_memcpy_to_gpu(void *dest, void *src, uint64_t size,
                       uint32_t gpu_index);

int cuda_memcpy_async_to_cpu(void *dest, const void *src, uint64_t size,
                             void *v_stream, uint32_t gpu_index);
int cuda_get_number_of_gpus();

int cuda_synchronize_device(uint32_t gpu_index);

int cuda_drop(void *ptr, uint32_t gpu_index);

int cuda_get_max_shared_memory(uint32_t gpu_index);
}
