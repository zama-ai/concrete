#include <gmp.h>
#include <libhcs/egcs.h>
#include "timing.h"

#define num_runs 100

int main(void)
{
    egcs_public_key *pk = egcs_init_public_key();
    egcs_private_key *vk = egcs_init_private_key();
    hcs_random *hr = hcs_init_random();
    egcs_generate_key_pair(pk, vk, hr, 2048);

    egcs_cipher *ca = egcs_init_cipher(),
                *cb = egcs_init_cipher(),
                *cc = egcs_init_cipher();

    mpz_t a, b, c;
    mpz_inits(a, b, c, NULL);

    mpz_set_ui(a, 4124124523);
    mpz_set_ui(b, 23423523);

    egcs_encrypt(pk, hr, ca, a);
    egcs_encrypt(pk, hr, cb, b);

    TIME_CODE(
#ifdef _OPENMP
            "Parallel",
#else
            "Single-core",
#endif
        for (int i = 0; i < num_runs; ++i)
            egcs_ee_mul(pk, cc, ca, cb);
    );
}
