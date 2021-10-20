#include "keyswitch.cuh"
#include "keyswitch.h"
#include "polynomial/parameters.cuh"

#include <cstdint>

/* Perform keyswitch on a batch of input LWE ciphertexts for 32 bits
 *
 *  - lwe_out: output batch of num_samples keyswitched ciphertexts c =
 * (a0,..an-1,b) where n is the LWE dimension
 *  - lwe_in: input batch of num_samples LWE ciphertexts, containing n
 *            mask values + 1 body value
 *
 * This function calls a wrapper to a device kernel that performs the keyswitch
 * 	- num_samples blocks of threads are launched
 */
void cuda_keyswitch_lwe_ciphertext_vector_32(void *v_stream, void *lwe_out, void *lwe_in,
                        void *ksk,
                        uint32_t lwe_dimension_before,
                        uint32_t lwe_dimension_after,
                        uint32_t base_log, uint32_t l_gadget,
                        uint32_t num_samples) {
    cuda_keyswitch_lwe_ciphertext_vector(
            v_stream, static_cast<uint32_t *>(lwe_out), static_cast<uint32_t *>(lwe_in),
            static_cast<uint32_t*>(ksk),
            lwe_dimension_before, lwe_dimension_after,
            base_log, l_gadget,
            num_samples);
}

/* Perform keyswitch on a batch of input LWE ciphertexts for 64 bits
 *
 *  - lwe_out: output batch of num_samples keyswitched ciphertexts c =
 * (a0,..an-1,b) where n is the LWE dimension
 *  - lwe_in: input batch of num_samples LWE ciphertexts, containing n
 *            mask values + 1 body value
 *
 * This function calls a wrapper to a device kernel that performs the keyswitch
 * 	- num_samples blocks of threads are launched
 */
void cuda_keyswitch_lwe_ciphertext_vector_64(void *v_stream, void *lwe_out, void *lwe_in,
                        void *ksk,
                        uint32_t lwe_dimension_before,
                        uint32_t lwe_dimension_after,
                        uint32_t base_log, uint32_t l_gadget,
                        uint32_t num_samples) {
    cuda_keyswitch_lwe_ciphertext_vector(
            v_stream, static_cast<uint64_t *>(lwe_out), static_cast<uint64_t *> (lwe_in),
            static_cast<uint64_t*>(ksk),
            lwe_dimension_before, lwe_dimension_after,
            base_log, l_gadget,
            num_samples);
}


