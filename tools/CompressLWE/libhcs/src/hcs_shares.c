#include <assert.h>
#include <stdlib.h>
#include <gmp.h>

#include "../include/libhcs/hcs_shares.h"

hcs_shares* hcs_init_shares(unsigned long size)
{
    hcs_shares *hs = malloc(sizeof(hcs_shares));
    if (!hs)
        goto failure;

    hs->shares = malloc(sizeof(mpz_t) * size);
    if (!hs->shares)
        goto failure;

    hs->flag = malloc(sizeof(int) * size);
    if (!hs->flag)
        goto failure;

    hs->size = size;

    for (unsigned long i = 0; i < hs->size; ++i)
        mpz_init(hs->shares[i]);

    return hs;

failure:
    if (hs->flag)
        free(hs->flag);
    if (hs->shares)
        free(hs->shares);
    if (hs)
        free(hs);

    return NULL;
}

void hcs_set_share(hcs_shares *hs, mpz_t share_value, unsigned long share_id)
{
    assert(share_id < hs->size);
    mpz_set(hs->shares[share_id], share_value);
    hs->flag[share_id] = 1;
}

void hcs_set_flag(hcs_shares *hs, unsigned long share_id)
{
    assert(share_id < hs->size);
    hs->flag[share_id] = 1;
}

void hcs_clear_flag(hcs_shares *hs, unsigned long share_id)
{
    assert(share_id < hs->size);
    hs->flag[share_id] = 0;
}

void hcs_toggle_flag(hcs_shares *hs, unsigned long share_id)
{
    assert(share_id < hs->size);
    hs->flag[share_id] = !hs->flag[share_id];
}

int hcs_tst_flag(hcs_shares *hs, unsigned long share_id)
{
    assert(share_id < hs->size);
    return hs->flag[share_id];
}

void hcs_free_shares(hcs_shares *hs)
{
    for (unsigned long i = 0; i < hs->size; ++i)
        mpz_clear(hs->shares[i]);

    free(hs->flag);
    free(hs->shares);
    free(hs);
}
