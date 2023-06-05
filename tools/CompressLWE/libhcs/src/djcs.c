/*
 * @file djcs.c
 *
 * Implementation of the Damgard-Jurik Cryptosystem (djcs).
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <gmp.h>

#include "../include/libhcs/hcs_random.h"
#include "../include/libhcs/djcs.h"
#include "com/parson.h"
#include "com/util.h"

/*
 * Algorithm as seen in the initial paper. Simple optimizations
 * have been added. rop and op can be aliases.
 */
static void dlog_s(djcs_private_key *vk, mpz_t rop, mpz_t op)
{
    mpz_t a, t1, t2, t3, kfact;
    mpz_inits(a, t1, t2, t3, kfact, NULL);

    /* Optimization: L(a mod n^(j+1)) = L(a mod n^(s+1)) mod n^j
     * where j <= s */
    mpz_mod(a, op, vk->n[vk->s]);
    mpz_sub_ui(a, a, 1);
    mpz_divexact(a, a, vk->n[0]);

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
            mpz_mul_ui(kfact, kfact, k);

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

djcs_public_key* djcs_init_public_key(void)
{
    djcs_public_key *pk = malloc(sizeof(djcs_public_key));
    if (!pk) return NULL;

    mpz_init(pk->g);
    pk->s = 0;
    pk->n = NULL;
    return pk;
}

djcs_private_key* djcs_init_private_key(void)
{
    djcs_private_key *vk = malloc(sizeof(djcs_private_key));
    if (!vk) return NULL;

    mpz_inits(vk->d, vk->mu, NULL);
    vk->s = 0;
    vk->n = NULL;
    return vk;
}

int djcs_generate_key_pair(djcs_public_key *pk, djcs_private_key *vk,
                           hcs_random *hr, unsigned long s, unsigned long bits)
{
    mpz_t p, q;

    pk->n = malloc(sizeof(mpz_t) * (s + 1));
    if (pk->n == NULL) return 1;
    vk->n = malloc(sizeof(mpz_t) * (s + 1));
    if (vk->n == NULL) return 1;

    pk->s = s;
    vk->s = s;

    mpz_init(p);
    mpz_init(q);
    mpz_init(pk->n[0]);
    mpz_init(vk->n[0]);

    mpz_random_prime(p, hr->rstate, 1 + (bits-1)/2);
    mpz_random_prime(q, hr->rstate, 1 + (bits-1)/2);
    mpz_mul(pk->n[0], p, q);
    mpz_sub_ui(vk->d, p, 1);
    mpz_sub_ui(q, q, 1);
    mpz_lcm(vk->d, vk->d, q);
    mpz_add_ui(q, q, 1);
    mpz_add_ui(pk->g, pk->n[0], 1);

    mpz_set(vk->n[0], pk->n[0]);
    for (unsigned long i = 1; i <= pk->s; ++i) {
        mpz_init_set(pk->n[i], pk->n[i-1]);
        mpz_mul(pk->n[i], pk->n[i], pk->n[0]);
        mpz_init_set(vk->n[i], pk->n[i]);
    }

    mpz_powm(vk->mu, pk->g, vk->d, vk->n[vk->s]);
    dlog_s(vk, vk->mu, vk->mu);
    mpz_invert(vk->mu, vk->mu, vk->n[vk->s-1]);

    mpz_clear(p);
    mpz_clear(q);

    return 0;
}

void djcs_encrypt(djcs_public_key *pk, hcs_random *hr, mpz_t rop, mpz_t plain1)
{
    mpz_t t1;
    mpz_init(t1);

    mpz_random_in_mult_group(t1, hr->rstate, pk->n[0]);
    mpz_powm(rop, pk->g, plain1, pk->n[pk->s]);
    mpz_powm(t1, t1, pk->n[pk->s-1], pk->n[pk->s]);
    mpz_mul(rop, rop, t1);
    mpz_mod(rop, rop, pk->n[pk->s]);

    mpz_clear(t1);
}

void djcs_reencrypt(djcs_public_key *pk, hcs_random *hr, mpz_t rop, mpz_t op)
{
    mpz_t t1;
    mpz_init(t1);

    mpz_random_in_mult_group(t1, hr->rstate, pk->n[0]);
    mpz_powm(t1, t1, pk->n[pk->s-1], pk->n[pk->s]);
    mpz_mul(rop, op, t1);
    mpz_mod(rop, rop, pk->n[pk->s]);

    mpz_clear(t1);
}

void djcs_ep_add(djcs_public_key *pk, mpz_t rop, mpz_t cipher1, mpz_t plain1)
{
    mpz_t t1;
    mpz_init(t1);

    mpz_set(t1, cipher1);
    mpz_powm(rop, pk->g, plain1, pk->n[pk->s]);
    mpz_mul(rop, rop, t1);
    mpz_mod(rop, rop, pk->n[pk->s]);

    mpz_clear(t1);
}

void djcs_ee_add(djcs_public_key *pk, mpz_t rop, mpz_t cipher1, mpz_t cipher2)
{
    mpz_mul(rop, cipher1, cipher2);
    mpz_mod(rop, rop, pk->n[pk->s]);
}

void djcs_ep_mul(djcs_public_key *pk, mpz_t rop, mpz_t cipher1, mpz_t plain1)
{
    mpz_powm(rop, cipher1, plain1, pk->n[pk->s]);
}

void djcs_decrypt(djcs_private_key *vk, mpz_t rop, mpz_t cipher1)
{
    mpz_powm(rop, cipher1, vk->d, vk->n[vk->s]);
    dlog_s(vk, rop, rop);
    mpz_mul(rop, rop, vk->mu);
    mpz_mod(rop, rop, vk->n[vk->s-1]);
}

void djcs_clear_public_key(djcs_public_key *pk)
{
    if (pk->n) {
        for (unsigned long i = 0; i < pk->s; ++i) {
            mpz_zero(pk->n[i]);
            mpz_clear(pk->n[i]);
        }
        free(pk->n);
    }

    mpz_zero(pk->g);
}

void djcs_clear_private_key(djcs_private_key *vk)
{
    if (vk->n) {
        for (unsigned long i = 0; i < vk->s; ++i) {
            mpz_zero(vk->n[i]);
            mpz_clear(vk->n[i]);
        }
        free(vk->n);
    }

    mpz_zero(vk->mu);
    mpz_zero(vk->d);
}

void djcs_free_public_key(djcs_public_key *pk)
{
    if (pk->n) {
        for (unsigned long i = 0; i <= pk->s; ++i) {
            mpz_zero(pk->n[i]);
            mpz_clear(pk->n[i]);
        }
        free(pk->n);
    }

    mpz_clear(pk->g);
    free(pk);
}

void djcs_free_private_key(djcs_private_key *vk)
{
    if (vk->n) {
        for (unsigned long i = 0; i <= vk->s; ++i) {
            mpz_zero(vk->n[i]);
            mpz_clear(vk->n[i]);
        }
        free(vk->n);
    }

    mpz_clear(vk->mu);
    mpz_clear(vk->d);
    free(vk);
}

int djcs_verify_key_pair(djcs_public_key *pk, djcs_private_key *vk)
{
    return (mpz_cmp(vk->n[0], pk->n[0]) == 0) && (pk->s == vk->s);
}

char *djcs_export_public_key(djcs_public_key *pk)
{
    char *buffer;
    char *retstr;

    JSON_Value *root = json_value_init_object();
    JSON_Object *obj  = json_value_get_object(root);
    buffer = mpz_get_str(NULL, HCS_INTERNAL_BASE, pk->n[0]);
    json_object_set_number(obj, "s", pk->s);
    json_object_set_string(obj, "n", buffer);
    retstr = json_serialize_to_string(root);

    json_value_free(root);
    free(buffer);
    return retstr;
}

char *djcs_export_private_key(djcs_private_key *vk)
{
    char *buffer;
    char *retstr;

    /* Allocate space for largest buffer output value used */
    size_t buffer_size = mpz_sizeinbase((mpz_cmp(vk->n[0], vk->d) >= 0
            ? vk->n[0] : vk->d), HCS_INTERNAL_BASE) + 2;
    buffer = malloc(buffer_size);

    JSON_Value *root = json_value_init_object();
    JSON_Object *obj = json_value_get_object(root);
    json_object_set_number(obj, "s", vk->s);
    mpz_get_str(buffer, HCS_INTERNAL_BASE, vk->n[0]);
    json_object_set_string(obj, "n", buffer);
    mpz_get_str(buffer, HCS_INTERNAL_BASE, vk->d);
    json_object_set_string(obj, "d", buffer);
    retstr = json_serialize_to_string(root);

    json_value_free(root);
    free(buffer);
    return retstr;
}

int djcs_import_public_key(djcs_public_key *pk, const char *json)
{
    JSON_Value *root = json_parse_string(json);
    JSON_Object *obj = json_value_get_object(root);
    pk->s = json_object_get_number(obj, "s");
    pk->n = malloc(sizeof(mpz_t) * (pk->s + 1));

    mpz_init(pk->n[0]);
    mpz_set_str(pk->n[0], json_object_get_string(obj, "n"), HCS_INTERNAL_BASE);
    json_value_free(root);

    /* Calculate remaining values */
    mpz_add_ui(pk->g, pk->n[0], 1);
    for (unsigned long i = 1; i <= pk->s; ++i) {
        mpz_init_set(pk->n[i], pk->n[i-1]);
        mpz_mul(pk->n[i], pk->n[i], pk->n[0]);
    }

    return 0;
}

int djcs_import_private_key(djcs_private_key *vk, const char *json)
{
    JSON_Value *root = json_parse_string(json);
    JSON_Object *obj = json_value_get_object(root);
    vk->s = json_object_get_number(obj, "s");
    vk->n = malloc(sizeof(mpz_t) * (vk->s + 1));

    mpz_init(vk->n[0]);
    mpz_set_str(vk->n[0], json_object_get_string(obj, "n"), HCS_INTERNAL_BASE);
    mpz_set_str(vk->d, json_object_get_string(obj, "d"), HCS_INTERNAL_BASE);
    json_value_free(root);

    /* Calculate remaining values */
    for (unsigned long i = 1; i <= vk->s; ++i) {
        mpz_init_set(vk->n[i], vk->n[i-1]);
        mpz_mul(vk->n[i], vk->n[i], vk->n[0]);
    }

    mpz_add_ui(vk->mu, vk->n[0], 1);
    mpz_powm(vk->mu, vk->mu, vk->d, vk->n[vk->s]);
    dlog_s(vk, vk->mu, vk->mu);
    mpz_invert(vk->mu, vk->mu, vk->n[vk->s-1]);
    return 0;
}
