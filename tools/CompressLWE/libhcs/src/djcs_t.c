/*
 * @file djcs_t.c
 * @author Marc Tiehuis
 * @date 15 March 2015
 *
 * Implementation of the Paillier Cryptosystem (djcs_t).
 *
 * This scheme is a threshold variant of the Paillier system. It loosely follows
 * the scheme presented in the paper by damgard-jurik, but with a chosen base of
 * 2, rather than the variable s+1. This scheme was written first for simplicity.
 *
 * POSSIBLE IMPROVEMENTS:
 *
 *  - Rewrite all assertions as checks that return error codes instead. We don't
 *    want to crash and instead want to relay this information to the caller.
 */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <gmp.h>

#include "../include/libhcs/hcs_random.h"
#include "../include/libhcs/djcs_t.h"
#include "com/util.h"

static void dlog_s(djcs_t_private_key *vk, mpz_t rop, mpz_t op)
{
    mpz_t a, t1, t2, t3, kfact;
    mpz_inits(a, t1, t2, t3, kfact, NULL);

    /* Optimization: L(a mod n^(j+1)) = L(a mod n^(s+1)) mod n^j
     * where j <= s */
    mpz_mod(a, op, vk->n[vk->s]);
    mpz_sub_ui(a, a, 1);
    mpz_divexact(a, a, vk->n[0]);

    /* rop and op can be aliased; store value in a */
    mpz_set_ui(rop, 0);

    for (unsigned long j = 1; j <= vk->s; ++j) {
        /* t1 = L(a mod n^j+1) */
        mpz_mod(t1, a, vk->n[j-1]);

        /* t2 = i */
        mpz_set(t2, rop);
        mpz_set_ui(kfact, 1);

        for (unsigned long k = 2; k <= j; ++k) {
            /* i = i - 1 */
            mpz_sub_ui(rop, rop, 1);
            mpz_mul_ui(kfact, kfact, k);    // iteratively compute k!

            /* t2 = t2 * i mod n^j */
            mpz_mul(t2, t2, rop);
            mpz_mod(t2, t2, vk->n[j-1]);

            /* t1 = t1 - (t2 * n^(k-1)) * k!^(-1)) mod n^j */
            mpz_invert(t3, kfact, vk->n[j-1]);
            mpz_mul(t3, t3, t2);
            mpz_mod(t3, t3, vk->n[j-1]);
            mpz_mul(t3, t3, vk->n[k-2]);
            mpz_mod(t3, t3, vk->n[j-1]);
            mpz_sub(t1, t1, t3);
            mpz_mod(t1, t1, vk->n[j-1]);
        }

        mpz_set(rop, t1);
    }

    mpz_zeros(a, t1, t2, t3, kfact, NULL);
    mpz_clears(a, t1, t2, t3, kfact, NULL);
}

#if 0
static int dlog_equality(mpz_t u, mpz_t uh, mpz_t v, mpz_t vh)
{
    /* Read up on the Fiat-Shamir heuristic and utilize a hash function
     * H. A hash function is needed anyway for dsa prime generation, so
     * put one in com */
    return 1;
}
#endif

djcs_t_public_key* djcs_t_init_public_key(void)
{
    djcs_t_public_key *pk = malloc(sizeof(djcs_t_public_key));
    if (!pk) return NULL;

    mpz_init(pk->g);
    return pk;
}

djcs_t_private_key* djcs_t_init_private_key(void)
{
    djcs_t_private_key *vk = malloc(sizeof(djcs_t_private_key));
    if (!vk) return NULL;

    vk->w = vk->l = vk->s = 0;
    mpz_inits(vk->p, vk->ph, vk->q, vk->qh,
             vk->v, vk->nsm, vk->m,
             vk->d, vk->delta, NULL);

    return vk;
}

/* Look into methods of using multiparty computation to generate these keys and
 * the data such that we don't have to have a trusted party for generation. */
void djcs_t_generate_key_pair(djcs_t_public_key *pk, djcs_t_private_key *vk, hcs_random *hr,
        const unsigned long s,
        const unsigned long bits, const unsigned long w, const unsigned long l)
{
    /* We can only perform this encryption if we have a w >= l / 2. Unsure
     * if this is the rounded value or not. i.e. is (1,3) system okay?
     * 3 / 2 = 1 by truncation >= 1. Need to confirm if this is allowed, or
     * more traditional rounding should be applied. */
    //assert(l / 2 <= w && w <= l);

    mpz_t t1, t2;
    mpz_init(t1);
    mpz_init(t2);

    /* Choose p and q to be safe primes */
    do {
        mpz_random_safe_prime(vk->p, vk->qh, hr->rstate, 1 + (bits-1)/2);
        mpz_random_safe_prime(vk->q, vk->ph, hr->rstate, 1 + (bits-1)/2);
    } while (mpz_cmp(vk->p, vk->q) == 0);

    pk->n = malloc(sizeof(mpz_t) * (s+1));
    vk->n = malloc(sizeof(mpz_t) * (s+1));
    pk->s = s;
    vk->s = s;

    /* n = p * q */
    mpz_init(pk->n[0]);
    mpz_mul(pk->n[0], vk->p, vk->q);

    for (unsigned long i = 1; i <= pk->s; ++i) {
        mpz_init(pk->n[i]);
        mpz_pow_ui(pk->n[i], pk->n[i-1], 2);
        mpz_init_set(vk->n[i], pk->n[i]);
    }

    /* g = n + 1 */
    mpz_add_ui(pk->g, pk->n[0], 1);

    /* Compute m = ph * qh */
    mpz_mul(vk->m, vk->ph, vk->qh);

    /* d == 1 mod n and d == 0 mod m */
    mpz_set_ui(t1, 1);
    mpz_set_ui(t2, 0);
    mpz_2crt(vk->d, t1, vk->n[vk->s-1], t2, vk->m);

    /* Compute n^s * m */
    mpz_mul(vk->nsm, vk->n[vk->s-1], vk->m);

    /* Set l and w in private key */
    vk->l = l;
    vk->w = w;

    /* Allocate space for verification values */
    vk->vi = malloc(sizeof(mpz_t) * l);
    for (unsigned long i = 0; i < l; ++i)
        mpz_init(vk->vi[i]);

    /* Precompute delta = l! */
    mpz_fac_ui(vk->delta, vk->l);

    /* Compute v being a cyclic generator of squares. This group is
     * always cyclic of order n * p' * q' since n is a safe prime product. */
    mpz_set(vk->v, vk->ph);

    mpz_clear(t1);
    mpz_clear(t2);
}

void djcs_t_encrypt(djcs_t_public_key *pk, hcs_random *hr, mpz_t rop, mpz_t plain1)
{
    mpz_t t1;
    mpz_init(t1);

    mpz_random_in_mult_group(t1, hr->rstate, pk->n[pk->s-1]);
    mpz_powm(rop, t1, pk->n[pk->s-1], pk->n[pk->s]);
    mpz_powm(t1, pk->g, plain1, pk->n[pk->s]);
    mpz_mul(rop, rop, t1);
    mpz_mod(rop, rop, pk->n[pk->s]);

    mpz_clear(t1);
}

void djcs_t_reencrypt(djcs_t_public_key *pk, hcs_random *hr, mpz_t rop, mpz_t op)
{
    mpz_t t1;
    mpz_init(t1);

    mpz_random_in_mult_group(t1, hr->rstate, pk->n[0]);
    mpz_powm(t1, t1, pk->n[pk->s-1], pk->n[pk->s]);
    mpz_mul(rop, op, t1);
    mpz_mod(rop, rop, pk->n[pk->s]);

    mpz_clear(t1);
}

void djcs_t_ep_add(djcs_t_public_key *pk, mpz_t rop, mpz_t cipher1, mpz_t plain1)
{
    mpz_t t1;
    mpz_init(t1);

    mpz_set(t1, cipher1);
    mpz_powm(rop, pk->g, plain1, pk->n[pk->s]);
    mpz_mul(rop, rop, t1);
    mpz_mod(rop, rop, pk->n[pk->s]);

    mpz_clear(t1);
}

void djcs_t_ee_add(djcs_t_public_key *pk, mpz_t rop, mpz_t cipher1, mpz_t cipher2)
{
    mpz_mul(rop, cipher1, cipher2);
    mpz_mod(rop, rop, pk->n[pk->s]);
}

void djcs_t_ep_mul(djcs_t_public_key *pk, mpz_t rop, mpz_t cipher1, mpz_t plain1)
{
    mpz_powm(rop, cipher1, plain1, pk->n[pk->s]);
}

mpz_t* djcs_t_init_polynomial(djcs_t_private_key *vk, hcs_random *hr)
{
    mpz_t *coeff = malloc(sizeof(mpz_t) * vk->w);
    if (coeff == NULL) return NULL;

    mpz_init_set(coeff[0], vk->d);
    for (unsigned long i = 1; i < vk->w; ++i) {
        mpz_init(coeff[i]);
        mpz_urandomm(coeff[i], hr->rstate, vk->nsm);
    }

    return coeff;
}

void djcs_t_compute_polynomial(djcs_t_private_key *vk, mpz_t *coeff, mpz_t rop, const unsigned long x)
{
    mpz_t t1, t2;
    mpz_init(t1);
    mpz_init(t2);

    /* Compute a polynomial with random coefficients in nm */
    mpz_set(rop, coeff[0]);
    for (unsigned long i = 1; i < vk->w; ++i) {
        mpz_ui_pow_ui(t1, x + 1, i); /* Correct for assumed 0-indexing of servers */
        mpz_mul(t1, t1, coeff[i]);
        mpz_add(rop, rop, t1);
        mpz_mod(rop, rop, vk->nsm);
    }

    mpz_clear(t1);
    mpz_clear(t2);
}

void djcs_t_free_polynomial(djcs_t_private_key *vk, mpz_t *coeff)
{
    for (unsigned long i = 0; i < vk->w; ++i) {
        mpz_clear(coeff[i]);
    }
    free(coeff);
}

djcs_t_auth_server* djcs_t_init_auth_server(void)
{
    djcs_t_auth_server *au = malloc(sizeof(djcs_t_auth_server));
    if (!au) return NULL;

    mpz_init(au->si);
    return au;
}

void djcs_t_set_auth_server(djcs_t_auth_server *au, mpz_t si, unsigned long i)
{
    mpz_set(au->si, si);
    au->i = i + 1; /* Assume 0-index and correct internally. */
}

void djcs_t_share_decrypt(djcs_t_private_key *vk, djcs_t_auth_server *au,
        mpz_t rop, mpz_t cipher1)
{
    mpz_t t1;
    mpz_init(t1);

    mpz_mul(t1, au->si, vk->delta);
    mpz_mul_ui(t1, t1, 2);
    mpz_powm(rop, cipher1, t1, vk->n[vk->s]);

    mpz_clear(t1);
}

void djcs_t_share_combine(djcs_t_private_key *vk, mpz_t rop, mpz_t *c)
{
    mpz_t t1, t2, t3;
    mpz_init(t1);
    mpz_init(t2);
    mpz_init(t3);

    /* Could alter loop to choose a random subset instead of always 0-indexing. */
    mpz_set_ui(rop, 1);
    for (unsigned long i = 0; i < vk->l; ++i) {

        if (mpz_cmp_ui(c[i], 0) == 0)
            continue; /* This share adds zero to the sum so skip. */

        /* Compute lambda_{0,i}^S. This is computed using the values of the
         * shares, not the indices? */
        mpz_set(t1, vk->delta);
        for (unsigned long j = 0; j < vk->l; ++j) {
            if ((j == i) || mpz_cmp_ui(c[j], 0) == 0)
                continue; /* i' in S\i and non-zero */

            long v = (long)j - (long)i;
            mpz_tdiv_q_ui(t1, t1, (v < 0 ? v*-1 : v));
            if (v < 0) mpz_neg(t1, t1);
            mpz_mul_ui(t1, t1, j + 1);
        }

        mpz_abs(t2, t1);
        mpz_mul_ui(t2, t2, 2);
        mpz_powm(t2, c[i], t2, vk->n[vk->s]);
        if (mpz_sgn(t1) < 0) mpz_invert(t2, t2, vk->n[vk->s]);
        mpz_mul(rop, rop, t2);
        mpz_mod(rop, rop, vk->n[vk->s]);
    }

    /* We now have c', so use algorithm from Theorem 1 to derive the result */
    dlog_s(vk, rop, rop);

    /* Multiply by (4*delta^2)^-1 mod n^2 to get result */
    mpz_pow_ui(t1, vk->delta, 2);
    mpz_mul_ui(t1, t1, 4);
    assert(mpz_invert(t1, t1, vk->n[vk->s-1])); // assume this inverse exists for now, add a check
    mpz_mul(rop, rop, t1);
    mpz_mod(rop, rop, vk->n[vk->s-1]);

    mpz_clear(t1);
    mpz_clear(t2);
    mpz_clear(t3);
}

void djcs_t_free_auth_server(djcs_t_auth_server *au)
{
    mpz_clear(au->si);
    free(au);
}

void djcs_t_clear_public_key(djcs_t_public_key *pk)
{
    mpz_zero(pk->g);
}

void djcs_t_clear_private_key(djcs_t_private_key *vk)
{
    mpz_zeros(vk->p, vk->ph, vk->q, vk->qh, vk->v, vk->nsm, vk->m,
              vk->d, vk->delta, NULL);

    if (vk->vi) {
        for (unsigned long i = 0; i < vk->l; ++i)
            mpz_clear(vk->vi[i]);
        free(vk->vi);
    }

    if (vk->n) {
        for (unsigned long i = 0; i <= vk->s; ++i)
            mpz_clear(vk->n[i]);
        free(vk->n);
    }
}

void djcs_t_free_public_key(djcs_t_public_key *pk)
{
    mpz_clear(pk->g);

    if (pk->n) {
        for (unsigned long i = 0; i <= pk->s; ++i)
            mpz_clear(pk->n[i]);
        free(pk->n);
    }

    free(pk);
}

void djcs_t_free_private_key(djcs_t_private_key *vk)
{
    mpz_clears(vk->p, vk->ph, vk->q, vk->qh, vk->v, vk->nsm, vk->m,
               vk->d, vk->delta, NULL);

    if (vk->vi) {
        for (unsigned long i = 0; i < vk->l; ++i)
            mpz_clear(vk->vi[i]);
        free(vk->vi);
    }

    if (vk->n) {
        for (unsigned long i = 0; i <= vk->s; ++i)
            mpz_clear(vk->n[i]);
        free(vk->n);
    }

    free(vk);
}
