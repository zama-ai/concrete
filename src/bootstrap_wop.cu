#include "bootstrap_wop.cuh"

void cuda_cmux_tree_32(
        void *v_stream,
        void *glwe_out,
        void *ggsw_in,
        void *lut_vector,
        uint32_t glwe_dimension,
        uint32_t polynomial_size,
        uint32_t base_log,
        uint32_t l_gadget,
        uint32_t r,
        uint32_t max_shared_memory) {

    switch (polynomial_size) {
        case 512:
            host_cmux_tree<uint32_t, int32_t, Degree<512>>(
                    v_stream,
                    (uint32_t *) glwe_out, (uint32_t *) ggsw_in, (uint32_t *) lut_vector,
                    glwe_dimension, polynomial_size, base_log, l_gadget, r,
                    max_shared_memory);
            break;
        case 1024:
            host_cmux_tree<uint32_t, int32_t, Degree<1024>>(
                    v_stream,
                    (uint32_t *) glwe_out, (uint32_t *) ggsw_in, (uint32_t *) lut_vector,
                    glwe_dimension, polynomial_size, base_log, l_gadget, r,
                    max_shared_memory);
            break;
        case 2048:
            host_cmux_tree<uint32_t, int32_t, Degree<2048>>(
                    v_stream,
                    (uint32_t *) glwe_out, (uint32_t *) ggsw_in, (uint32_t *) lut_vector,
                    glwe_dimension, polynomial_size, base_log, l_gadget, r,
                    max_shared_memory);
            break;
        case 4096:
            host_cmux_tree<uint32_t, int32_t, Degree<4096>>(
                    v_stream,
                    (uint32_t *) glwe_out, (uint32_t *) ggsw_in, (uint32_t *) lut_vector,
                    glwe_dimension, polynomial_size, base_log, l_gadget, r,
                    max_shared_memory);
            break;
        case 8192:
            host_cmux_tree<uint32_t, int32_t, Degree<8192>>(
                    v_stream,
                    (uint32_t *) glwe_out, (uint32_t *) ggsw_in, (uint32_t *) lut_vector,
                    glwe_dimension, polynomial_size, base_log, l_gadget, r,
                    max_shared_memory);
            break;
    }
}

void cuda_cmux_tree_64(
        void *v_stream,
        void *glwe_out,
        void *ggsw_in,
        void *lut_vector,
        uint32_t glwe_dimension,
        uint32_t polynomial_size,
        uint32_t base_log,
        uint32_t l_gadget,
        uint32_t r,
        uint32_t max_shared_memory) {

    switch (polynomial_size) {
        case 512:
            host_cmux_tree<uint64_t, int64_t, Degree<512>>(
                    v_stream,
                    (uint64_t *) glwe_out, (uint64_t *) ggsw_in,(uint64_t *) lut_vector,
                    glwe_dimension, polynomial_size, base_log, l_gadget, r,
                    max_shared_memory);
            break;
        case 1024:
            host_cmux_tree<uint64_t, int64_t, Degree<1024>>(
                    v_stream,
                    (uint64_t *) glwe_out, (uint64_t *) ggsw_in,(uint64_t *) lut_vector,
                    glwe_dimension, polynomial_size, base_log, l_gadget, r,
                    max_shared_memory);
            break;
        case 2048:
            host_cmux_tree<uint64_t, int64_t, Degree<2048>>(
                    v_stream,
                    (uint64_t *) glwe_out, (uint64_t *) ggsw_in,(uint64_t *) lut_vector,
                    glwe_dimension, polynomial_size, base_log, l_gadget, r,
                    max_shared_memory);
            break;
        case 4096:
            host_cmux_tree<uint64_t, int64_t, Degree<4096>>(
                    v_stream,
                    (uint64_t *) glwe_out, (uint64_t *) ggsw_in,(uint64_t *) lut_vector,
                    glwe_dimension, polynomial_size, base_log, l_gadget, r,
                    max_shared_memory);
            break;
        case 8192:
            host_cmux_tree<uint64_t, int64_t, Degree<8192>>(
                    v_stream,
                    (uint64_t *) glwe_out, (uint64_t *) ggsw_in,(uint64_t *) lut_vector,
                    glwe_dimension, polynomial_size, base_log, l_gadget, r,
                    max_shared_memory);
            break;
    }
}


void cuda_extract_bits_32(
    void *v_stream,
    void *list_lwe_out,
    void *lwe_in,
    void *lwe_in_buffer,
    void *lwe_in_shifted_buffer,
    void *lwe_out_ks_buffer,
    void *lwe_out_pbs_buffer,
    void *lut_pbs,
    void *lut_vector_indexes,
    void *ksk,
    void *fourier_bsk,
    uint32_t number_of_bits,
    uint32_t delta_log,
    uint32_t lwe_dimension_before,
    uint32_t lwe_dimension_after,
    uint32_t base_log_bsk,
    uint32_t l_gadget_bsk,
    uint32_t base_log_ksk,
    uint32_t l_gadget_ksk,
    uint32_t number_of_samples)
{
  switch (lwe_dimension_before) {
  case 512:
    host_extract_bits<uint32_t, Degree<512>>(
        v_stream, (uint32_t *)list_lwe_out, (uint32_t *)lwe_in,
        (uint32_t *)lwe_in_buffer, (uint32_t *)lwe_in_shifted_buffer,
        (uint32_t *)lwe_out_ks_buffer, (uint32_t *)lwe_out_pbs_buffer,
        (uint32_t *)lut_pbs, (uint32_t *)lut_vector_indexes, (uint32_t *)ksk,
        (double2 *)fourier_bsk, number_of_bits, delta_log,
        lwe_dimension_before, lwe_dimension_after, base_log_bsk, l_gadget_bsk,
        base_log_ksk, l_gadget_ksk, number_of_samples);
    break;
  case 1024:
    host_extract_bits<uint32_t, Degree<1024>>(
        v_stream, (uint32_t *)list_lwe_out, (uint32_t *)lwe_in,
        (uint32_t *)lwe_in_buffer, (uint32_t *)lwe_in_shifted_buffer,
        (uint32_t *)lwe_out_ks_buffer, (uint32_t *)lwe_out_pbs_buffer,
        (uint32_t *)lut_pbs, (uint32_t *)lut_vector_indexes, (uint32_t *)ksk,
        (double2 *)fourier_bsk, number_of_bits, delta_log,
        lwe_dimension_before, lwe_dimension_after, base_log_bsk, l_gadget_bsk,
        base_log_ksk, l_gadget_ksk, number_of_samples);
    break;
  case 2048:
    host_extract_bits<uint32_t, Degree<2048>>(
        v_stream, (uint32_t *)list_lwe_out, (uint32_t *)lwe_in,
        (uint32_t *)lwe_in_buffer, (uint32_t *)lwe_in_shifted_buffer,
        (uint32_t *)lwe_out_ks_buffer, (uint32_t *)lwe_out_pbs_buffer,
        (uint32_t *)lut_pbs, (uint32_t *)lut_vector_indexes, (uint32_t *)ksk,
        (double2 *)fourier_bsk, number_of_bits, delta_log,
        lwe_dimension_before, lwe_dimension_after, base_log_bsk, l_gadget_bsk,
        base_log_ksk, l_gadget_ksk, number_of_samples);
    break;
  default:
    break;
  }

}



void cuda_extract_bits_64(
    void *v_stream,
    void *list_lwe_out,
    void *lwe_in,
    void *lwe_in_buffer,
    void *lwe_in_shifted_buffer,
    void *lwe_out_ks_buffer,
    void *lwe_out_pbs_buffer,
    void *lut_pbs,
    void *lut_vector_indexes,
    void *ksk,
    void *fourier_bsk,
    uint32_t number_of_bits,
    uint32_t delta_log,
    uint32_t lwe_dimension_before,
    uint32_t lwe_dimension_after,
    uint32_t base_log_bsk,
    uint32_t l_gadget_bsk,
    uint32_t base_log_ksk,
    uint32_t l_gadget_ksk,
    uint32_t number_of_samples)
{
  switch (lwe_dimension_before) {
  case 512:
    host_extract_bits<uint64_t, Degree<512>>(
        v_stream, (uint64_t *)list_lwe_out, (uint64_t *)lwe_in,
        (uint64_t *)lwe_in_buffer, (uint64_t *)lwe_in_shifted_buffer,
        (uint64_t *)lwe_out_ks_buffer, (uint64_t *)lwe_out_pbs_buffer,
        (uint64_t *)lut_pbs, (uint32_t *)lut_vector_indexes, (uint64_t *)ksk,
        (double2 *)fourier_bsk, number_of_bits, delta_log,
        lwe_dimension_before, lwe_dimension_after, base_log_bsk, l_gadget_bsk,
        base_log_ksk, l_gadget_ksk, number_of_samples);
    break;
  case 1024:
    host_extract_bits<uint64_t, Degree<1024>>(
        v_stream, (uint64_t *)list_lwe_out, (uint64_t *)lwe_in,
        (uint64_t *)lwe_in_buffer, (uint64_t *)lwe_in_shifted_buffer,
        (uint64_t *)lwe_out_ks_buffer, (uint64_t *)lwe_out_pbs_buffer,
        (uint64_t *)lut_pbs, (uint32_t *)lut_vector_indexes, (uint64_t *)ksk,
        (double2 *)fourier_bsk, number_of_bits, delta_log,
        lwe_dimension_before, lwe_dimension_after, base_log_bsk, l_gadget_bsk,
        base_log_ksk, l_gadget_ksk, number_of_samples);
    break;
  case 2048:
    host_extract_bits<uint64_t, Degree<2048>>(
        v_stream, (uint64_t *)list_lwe_out, (uint64_t *)lwe_in,
        (uint64_t *)lwe_in_buffer, (uint64_t *)lwe_in_shifted_buffer,
        (uint64_t *)lwe_out_ks_buffer, (uint64_t *)lwe_out_pbs_buffer,
        (uint64_t *)lut_pbs, (uint32_t *)lut_vector_indexes, (uint64_t *)ksk,
        (double2 *)fourier_bsk, number_of_bits, delta_log,
        lwe_dimension_before, lwe_dimension_after, base_log_bsk, l_gadget_bsk,
        base_log_ksk, l_gadget_ksk, number_of_samples);
    break;
  default:
    break;
  }

}


