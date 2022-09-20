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