#include <gmp.h>
#include <libhcs/pcs.h>
#include "chrono.h"

int main(void)
{
#define test_vector_size 5
    int test_vector[test_vector_size][2] = {
        /* num_runs, key_size */
        { 10000, 256 },
        { 10000, 512 },
        { 2000, 1024 },
        { 200,  2048 },
        { 25,   4096 }
    };

    pcs_public_key *pk = pcs_init_public_key();
    pcs_private_key *vk = pcs_init_private_key();
    hcs_random *hr = hcs_init_random();

    const char *core_string =
#ifdef _OPENMP
            "Parallel"
#else
            "Single-core"
#endif
            ;

    mpz_t a, b, c, d;
    mpz_inits(a, b, c, d, NULL);

    for (int i = 0; i < test_vector_size; ++i) {
        double total = 0;
        chrono timer;

        mpz_set_ui(a, 4124124523);
        mpz_set_ui(b, 23423508023);
        mpz_set_ui(d, 1);
        pcs_generate_key_pair(pk, vk, hr, test_vector[i][1]);
        pcs_encrypt(pk, hr, a, a);

        for (int j = 0; j < test_vector[i][0]; ++j) {
            chrono_start(&timer);
            pcs_decrypt(vk, c, a);
            chrono_end(&timer);
            total += chrono_get_msec(&timer);
            pcs_ep_add(pk, a, a, d);
        }

        printf("%s: (%d): %.15f\n", core_string, test_vector[i][1],
                total / test_vector[i][0]);
    }
}
