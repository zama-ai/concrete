/**
 * @file hcs_random.c
 *
 * Provides easy to use random state functions, that are required
 * for all probablistic schemes.
 *
 * This is effectively a wrapper around a gmp_randstate_t type. However, we
 * provide safe seeding of the generator, instead of the user having to.
 * This is provided as a new struct to more closely resemble the usage of the
 * other types within this library.
 */

#include <gmp.h>

#include "../include/libhcs/hcs_random.h"
#include "com/util.h"

/* Currently one can set the seed. This is used only for testing and will
   be altered at a latter time to take no arguments. */
hcs_random* hcs_init_random(void)
{
    hcs_random *hr = malloc(sizeof(hcs_random));
    if (hr == NULL) return NULL;

    mpz_t t1;
    mpz_init_set_ui(t1, 0);
    gmp_randinit_default(hr->rstate);
#ifndef HCS_STATIC_SEED // Comment out to zero seed for testing
    mpz_seed(t1, HCS_RAND_SEED_BITS);
#endif
    gmp_randseed(hr->rstate, t1);

    mpz_clear(t1);
    return hr;
}

int hcs_reseed_random(hcs_random *hr)
{
    // Currently assume we get values correctly. We should check that we
    // read correctly and alter mpz_seed to return a status code.
    mpz_t t1;
    mpz_init(t1);

#ifndef HCS_STATIC_SEED
    if (mpz_seed(t1, HCS_RAND_SEED_BITS) != HCS_OK) {
        mpz_clear(t1);
        return 0;
    }
#endif

    gmp_randseed(hr->rstate, t1);
    mpz_clear(t1);
    return 1;
}

void hcs_free_random(hcs_random *hr)
{
    gmp_randclear(hr->rstate);
    free(hr);
}
