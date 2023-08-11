#include <benchmark/benchmark.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <setup_and_teardown.h>
#include <omp.h>

const bool USE_MULTI_GPU = false;

typedef struct {
    int lwe_dimension;
    int glwe_dimension;
    int polynomial_size;
    double lwe_modular_variance;
    double glwe_modular_variance;
    int pbs_base_log;
    int pbs_level;
    int ksk_base_log;
    int ksk_level;
    int total_message_bits;
    int number_of_blocks;
    int message_modulus;
    int carry_modulus;
    PBS_TYPE pbs_type;
} IntegerMultiplicationBenchmarkParams;

class IntegerMultiplication_u64 : public benchmark::Fixture {
protected:
    int lwe_dimension;
    int glwe_dimension;
    int polynomial_size;
    double lwe_modular_variance = 4.478453795193731e-11;
    double glwe_modular_variance = 8.645717832544903e-32;
    int pbs_base_log;
    int pbs_level;
    int ksk_base_log;
    int ksk_level;
    int message_modulus;
    int carry_modulus;
    int total_message_bits;
    int number_of_blocks;
    int payload_modulus;
    PBS_TYPE pbs_type;
    uint64_t delta;

    std::vector<void *> d_bsk_array;
    std::vector<uint64_t *> d_ksk_array;
    std::vector<uint64_t *> d_lwe_ct_in_array_1;
    std::vector<uint64_t *> d_lwe_ct_in_array_2;
    std::vector<uint64_t *> d_lwe_ct_out_array;
    uint64_t *lwe_sk_in;
    uint64_t *lwe_sk_out;
    uint64_t *plaintexts_1;
    uint64_t *plaintexts_2;
    std::vector<int_mul_memory<uint64_t> *> mem_ptr_array;

    Csprng *csprng;
    int max_gpus_to_use;
    int operations_per_gpu;

    int num_gpus;

public:
    void SetUp(const ::benchmark::State &state) {
        cudaDeviceSynchronize();

        lwe_dimension = state.range(0);
        glwe_dimension = state.range(1);
        polynomial_size = state.range(2);
        lwe_modular_variance = state.range(3);
        glwe_modular_variance = state.range(4);
        pbs_base_log = state.range(5);
        pbs_level = state.range(6);
        ksk_base_log = state.range(7);
        ksk_level = state.range(8);
        total_message_bits = state.range(9);
        number_of_blocks = state.range(10);
        message_modulus = state.range(11);
        carry_modulus = state.range(12);
        int pbs_type_int = state.range(13);
        max_gpus_to_use = state.range(14);
        operations_per_gpu = state.range(15);

        pbs_type = static_cast<PBS_TYPE>(pbs_type_int);

        num_gpus = std::min(cuda_get_number_of_gpus(), max_gpus_to_use);

        for (int device = 0; device < num_gpus; device++) {
            cudaSetDevice(device);
            cudaStream_t *stream = cuda_create_stream(device);

            void *d_bsk_array_per_gpu;
            uint64_t *d_ksk_array_per_gpu;
            uint64_t *d_lwe_ct_in_array_1_per_gpu;
            uint64_t *d_lwe_ct_in_array_2_per_gpu;
            uint64_t *d_lwe_ct_out_array_per_gpu;
            int_mul_memory<uint64_t> *mem_ptr_per_gpu = new int_mul_memory<uint64_t>;

            integer_multiplication_setup(
                    stream, &csprng, &lwe_sk_in, &lwe_sk_out,
                    &d_bsk_array_per_gpu, &d_ksk_array_per_gpu,
                    &plaintexts_1, &plaintexts_2, &d_lwe_ct_in_array_1_per_gpu,
                    &d_lwe_ct_in_array_2_per_gpu, &d_lwe_ct_out_array_per_gpu,
                    mem_ptr_per_gpu, lwe_dimension, glwe_dimension, polynomial_size,
                    lwe_modular_variance, glwe_modular_variance, pbs_base_log, pbs_level,
                    ksk_base_log, ksk_level, total_message_bits, number_of_blocks,
                    message_modulus, carry_modulus, &delta, 1, 1, pbs_type, device);

            if (USE_MULTI_GPU) {
                scratch_cuda_integer_mult_radix_ciphertext_kb_64_multi_gpu(
                        mem_ptr_per_gpu, d_bsk_array_per_gpu, d_ksk_array_per_gpu,
                        message_modulus, carry_modulus, glwe_dimension, lwe_dimension,
                        polynomial_size, pbs_base_log, pbs_level, ksk_base_log, ksk_level,
                        number_of_blocks, pbs_type, cuda_get_max_shared_memory(device),
                        true);

            } else {
                scratch_cuda_integer_mult_radix_ciphertext_kb_64(
                        stream, device, (void *)mem_ptr_per_gpu, message_modulus,
                        carry_modulus, glwe_dimension, lwe_dimension, polynomial_size,
                        pbs_base_log, pbs_level, ksk_base_log, ksk_level, number_of_blocks,
                        pbs_type, cuda_get_max_shared_memory(device), true);
            }

            d_bsk_array.push_back(d_bsk_array_per_gpu);
            d_ksk_array.push_back(d_ksk_array_per_gpu);
            d_lwe_ct_in_array_1.push_back(d_lwe_ct_in_array_1_per_gpu);
            d_lwe_ct_in_array_2.push_back(d_lwe_ct_in_array_2_per_gpu);
            d_lwe_ct_out_array.push_back(d_lwe_ct_out_array_per_gpu);
            mem_ptr_array.push_back(mem_ptr_per_gpu);

            cuda_synchronize_stream(stream);
            cuda_destroy_stream(stream, device);
        }
    }

    void TearDown(const ::benchmark::State &state) {
        cudaDeviceSynchronize();
        concrete_cpu_destroy_concrete_csprng(csprng);
        free(csprng);
        free(lwe_sk_in);
        free(lwe_sk_out);
        free(plaintexts_1);
        free(plaintexts_2);

        for (int device = 0; device < num_gpus; device++) {
            cudaSetDevice(device);
            cudaStream_t *stream = cuda_create_stream(device);
            cuda_drop_async(d_bsk_array[device], stream, device);
            cuda_drop_async(d_ksk_array[device], stream, device);
            cuda_drop_async(d_lwe_ct_in_array_1[device], stream, device);
            cuda_drop_async(d_lwe_ct_in_array_2[device], stream, device);
            cuda_drop_async(d_lwe_ct_out_array[device], stream, device);

            int_mul_memory<uint64_t> *mem_ptr = mem_ptr_array[device];

            cuda_drop_async(mem_ptr->vector_result_sb, stream, 0);
            cuda_drop_async(mem_ptr->block_mul_res, stream, 0);
            cuda_drop_async(mem_ptr->small_lwe_vector, stream, 0);
            cuda_drop_async(mem_ptr->lwe_pbs_out_array, stream, 0);
            cuda_drop_async(mem_ptr->test_vector_array, stream, 0);
            cuda_drop_async(mem_ptr->message_acc, stream, 0);
            cuda_drop_async(mem_ptr->carry_acc, stream, 0);
            cuda_drop_async(mem_ptr->test_vector_indexes, stream, 0);
            cuda_drop_async(mem_ptr->tvi_message, stream, 0);
            cuda_drop_async(mem_ptr->tvi_carry, stream, 0);
            cuda_drop_async(mem_ptr->pbs_buffer, stream, 0);
            for (int i = 0; i < mem_ptr->p2p_gpu_count; i++) {
                cuda_drop_async(mem_ptr->device_to_device_buffer[i], mem_ptr->streams[i],
                                i);
                cuda_drop_async(mem_ptr->pbs_buffer_multi_gpu[i], mem_ptr->streams[i], i);
                cuda_drop_async(mem_ptr->pbs_input_multi_gpu[i], mem_ptr->streams[i], i);
                cuda_drop_async(mem_ptr->pbs_output_multi_gpu[i], mem_ptr->streams[i], i);
                cuda_drop_async(mem_ptr->test_vector_multi_gpu[i], mem_ptr->streams[i], i);
                cuda_drop_async(mem_ptr->tvi_lsb_multi_gpu[i], mem_ptr->streams[i], i);
                cuda_drop_async(mem_ptr->tvi_msb_multi_gpu[i], mem_ptr->streams[i], i);
                cuda_drop_async(mem_ptr->tvi_message_multi_gpu[i], mem_ptr->streams[i], i);
                cuda_drop_async(mem_ptr->tvi_carry_multi_gpu[i], mem_ptr->streams[i], i);
                if (i) {
                    cuda_drop_async(mem_ptr->bsk_multi_gpu[i], mem_ptr->streams[i], i);
                    cuda_drop_async(mem_ptr->ksk_multi_gpu[i], mem_ptr->streams[i], i);
                }
                cuda_destroy_stream(mem_ptr->streams[i], i);
            }

            cuda_synchronize_stream(stream);
            cuda_destroy_stream(stream, device);
        }

        d_bsk_array.clear();
        d_ksk_array.clear();
        d_lwe_ct_in_array_1.clear();
        d_lwe_ct_in_array_2.clear();
        d_lwe_ct_out_array.clear();
        mem_ptr_array.clear();
        cudaDeviceReset();
    }
};

BENCHMARK_DEFINE_F(IntegerMultiplication_u64,
        ConcreteCuda_IntegerMultiplication)
(benchmark::State &st) {
int8_t *mult_buffer;
uint32_t ct_degree_out = 0;
uint32_t ct_degree_left = 0;
uint32_t ct_degree_right = 0;

omp_set_nested(true);

for (auto _ : st) {
// Execute multiplication
#pragma omp parallel for num_threads(num_gpus)
for (int device = 0; device < num_gpus; device++) {
cudaSetDevice(device);

auto d_lwe_ct_out = d_lwe_ct_out_array[device];
auto d_lwe_ct_in_1 = d_lwe_ct_in_array_1[device];
auto d_lwe_ct_in_2 = d_lwe_ct_in_array_2[device];
auto d_bsk = d_bsk_array[device];
auto d_ksk = d_ksk_array[device];
auto mem_ptr = mem_ptr_array[device];

#pragma omp parallel for num_threads(operations_per_gpu)
for (int i = 0; i < operations_per_gpu; i++) {
cudaStream_t *stream = cuda_create_stream(device);
if (USE_MULTI_GPU) {
cuda_integer_mult_radix_ciphertext_kb_64_multi_gpu(
(void *)d_lwe_ct_out, (void *)d_lwe_ct_in_1,
(void *)d_lwe_ct_in_2, &ct_degree_out, &ct_degree_left,
&ct_degree_right, d_bsk, d_ksk, (void *)mem_ptr, message_modulus,
carry_modulus, glwe_dimension, lwe_dimension, polynomial_size,
pbs_base_log, pbs_level, ksk_base_log, ksk_level,
number_of_blocks, pbs_type, cuda_get_max_shared_memory(device));
} else {
cuda_integer_mult_radix_ciphertext_kb_64(
        stream, device, (void *)d_lwe_ct_out, (void *)d_lwe_ct_in_1,
(void *)d_lwe_ct_in_2, &ct_degree_out, &ct_degree_left,
&ct_degree_right, d_bsk, d_ksk, (void *)mem_ptr, message_modulus,
carry_modulus, glwe_dimension, lwe_dimension, polynomial_size,
pbs_base_log, pbs_level, ksk_base_log, ksk_level,
number_of_blocks, pbs_type, cuda_get_max_shared_memory(device));
}
cuda_synchronize_stream(stream);
cuda_destroy_stream(stream, device);
}
}
}
}

static void IntegerMultiplicationBenchmarkGenerateParams(
        benchmark::internal::Benchmark *b) {
    // Define the parameters to benchmark
    std::vector<IntegerMultiplicationBenchmarkParams> params = {
            (IntegerMultiplicationBenchmarkParams){
                    744, 1, 2048, 4.478453795193731e-11, 8.645717832544903e-32, 23, 1, 3,
                    5, 8, 4, 4, 4, LOW_LAT},
            (IntegerMultiplicationBenchmarkParams){
                    744, 1, 2048, 4.478453795193731e-11, 8.645717832544903e-32, 23, 1, 3,
                    5, 16, 8, 4, 4, LOW_LAT},
            (IntegerMultiplicationBenchmarkParams){
                    744, 1, 2048, 4.478453795193731e-11, 8.645717832544903e-32, 23, 1, 3,
                    5, 32, 16, 4, 4, LOW_LAT},
            (IntegerMultiplicationBenchmarkParams){
                    744, 1, 2048, 4.478453795193731e-11, 8.645717832544903e-32, 23, 1, 3,
                    5, 40, 20, 4, 4, LOW_LAT},
            (IntegerMultiplicationBenchmarkParams){
                    744, 1, 2048, 4.478453795193731e-11, 8.645717832544903e-32, 23, 1, 3,
                    5, 64, 32, 4, 4, LOW_LAT},
            (IntegerMultiplicationBenchmarkParams){
                    744, 1, 2048, 4.478453795193731e-11, 8.645717832544903e-32, 23, 1, 3,
                    5, 128, 64, 4, 4, LOW_LAT},
            (IntegerMultiplicationBenchmarkParams){
                    744, 1, 2048, 4.478453795193731e-11, 8.645717832544903e-32, 23, 1, 3,
                    5, 256, 128, 4, 4, LOW_LAT},
            (IntegerMultiplicationBenchmarkParams){
                    744, 1, 2048, 4.478453795193731e-11, 8.645717832544903e-32, 23, 1, 3,
                    5, 8, 4, 4, 4, MULTI_BIT},
            (IntegerMultiplicationBenchmarkParams){
                    744, 1, 2048, 4.478453795193731e-11, 8.645717832544903e-32, 23, 1, 3,
                    5, 16, 8, 4, 4, MULTI_BIT},
            (IntegerMultiplicationBenchmarkParams){
                    744, 1, 2048, 4.478453795193731e-11, 8.645717832544903e-32, 23, 1, 3,
                    5, 32, 16, 4, 4, MULTI_BIT},
            (IntegerMultiplicationBenchmarkParams){
                    744, 1, 2048, 4.478453795193731e-11, 8.645717832544903e-32, 23, 1, 3,
                    5, 40, 20, 4, 4, MULTI_BIT},
            (IntegerMultiplicationBenchmarkParams){
                    744, 1, 2048, 4.478453795193731e-11, 8.645717832544903e-32, 23, 1, 3,
                    5, 64, 32, 4, 4, MULTI_BIT},
            (IntegerMultiplicationBenchmarkParams){
                    744, 1, 2048, 4.478453795193731e-11, 8.645717832544903e-32, 23, 1, 3,
                    5, 128, 64, 4, 4, MULTI_BIT},
            (IntegerMultiplicationBenchmarkParams){
                    744, 1, 2048, 4.478453795193731e-11, 8.645717832544903e-32, 23, 1, 3,
                    5, 256, 128, 4, 4, MULTI_BIT},
    };

    int max_gpus_to_use = 8;

    // Add to the list of parameters to benchmark
    for(int operations_per_gpu = 1; operations_per_gpu < 10; operations_per_gpu++)
        for (auto x : params) {
            b->Args({x.lwe_dimension, x.glwe_dimension, x.polynomial_size,
                     x.lwe_modular_variance, x.glwe_modular_variance, x.pbs_base_log,
                     x.pbs_level, x.ksk_base_log, x.ksk_level, x.total_message_bits,
                     x.number_of_blocks, x.message_modulus, x.carry_modulus,
                     x.pbs_type, max_gpus_to_use, operations_per_gpu});
        }
}

BENCHMARK_REGISTER_F(IntegerMultiplication_u64,
        ConcreteCuda_IntegerMultiplication)
->Apply(IntegerMultiplicationBenchmarkGenerateParams);
