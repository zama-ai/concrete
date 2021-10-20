#ifndef CNCRT_KS_H_
#define CNCRT_KS_H_

#include <cstdint>

extern "C" {

void cuda_keyswitch_lwe_ciphertext_vector_32(void *v_stream, void *lwe_out, void *lwe_in,
                        void *ksk,
                        uint32_t lwe_dimension_before,
                        uint32_t lwe_dimension_after,
                        uint32_t base_log, uint32_t l_gadget,
                        uint32_t num_samples);

void cuda_keyswitch_lwe_ciphertext_vector_64(void *v_stream, void *lwe_out, void *lwe_in,
                        void *ksk,
                        uint32_t lwe_dimension_before,
                        uint32_t lwe_dimension_after,
                        uint32_t base_log, uint32_t l_gadget,
                        uint32_t num_samples);

}

#endif // CNCRT_KS_H_
