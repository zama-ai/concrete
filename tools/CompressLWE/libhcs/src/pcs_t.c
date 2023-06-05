/*
 * @file pcs_t.c
 *
 * Implementation of the Paillier Cryptosystem (pcs_t).
 *
 * This scheme is a threshold variant of the Paillier system. It loosely
 * follows the scheme presented in the paper by damgard-jurik, but with a
 * chosen base of 2, rather than the variable s+1. This scheme was written
 * first for simplicity.
 *
 * @todo Desperately need to move away from naive prime generation here, as
 * it is currently a massive bottleneck and computing large 2048 bit safe
 * primes is taking to long.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <gmp.h>

#include "../include/libhcs/hcs_random.h"
#include "../include/libhcs/hcs_shares.h"
#include "../include/libhcs/pcs_t.h"
#include "com/parson.h"
#include "com/util.h"

#define HCS_HASH_SIZE 160

/* This is simply L(x) when s = 1 */
static void dlog_s(mpz_t n, mpz_t rop, mpz_t op)
{
    mpz_sub_ui(rop, op, 1);
    mpz_divexact(rop, rop, n);
    mpz_mod(rop, rop, n);
}

pcs_t_public_key* pcs_t_init_public_key(void)
{
    pcs_t_public_key *pk = malloc(sizeof(pcs_t_public_key));
    if (!pk) return NULL;

    mpz_init(pk->n);
    mpz_init(pk->n2);
    mpz_init(pk->g);
    mpz_init(pk->delta);
    return pk;
}

pcs_t_private_key* pcs_t_init_private_key(void)
{
    pcs_t_private_key *vk = malloc(sizeof(pcs_t_private_key));
    if (!vk) return NULL;

    vk->w = vk->l = 0;
    mpz_init(vk->v);
    mpz_init(vk->nm);
    mpz_init(vk->n);
    mpz_init(vk->n2);
    mpz_init(vk->d);
    return vk;
}

/* Look into methods of using multiparty computation to generate these keys
 * and the data so we don't have to have a trusted party for generation. */
int pcs_t_generate_key_pair(pcs_t_public_key *pk, pcs_t_private_key *vk,
        hcs_random *hr, const unsigned long bits, const unsigned long w,
        const unsigned long l)
{
    /* The paper does describe some bounds on w, l */
    //assert(l / 2 <= w && w <= l);

    vk->vi = malloc(sizeof(mpz_t) * l);
    if (vk->vi == NULL) return 0;

    mpz_t t1, t2, t3, t4;
    mpz_init(t1);
    mpz_init(t2);
    mpz_init(t3);
    mpz_init(t4);

    do {
        mpz_random_safe_prime(t1, t2, hr->rstate, 1 + (bits-1)/2);
        mpz_random_safe_prime(t3, t4, hr->rstate, 1 + (bits-1)/2);
    } while (mpz_cmp(t1, t3) == 0);

    mpz_mul(pk->n, t1, t3);
    mpz_set(vk->n, pk->n);
    mpz_pow_ui(pk->n2, pk->n, 2);
    mpz_set(vk->n2, pk->n2);
    mpz_add_ui(pk->g, pk->n, 1);
    mpz_mul(t3, t2, t4);
    mpz_mul(vk->nm, vk->n, t3);
    mpz_set_ui(t1, 1);
    mpz_set_ui(t2, 0);
    mpz_2crt(vk->d, t1, vk->n, t2, t3);
    mpz_fac_ui(pk->delta, l);

    vk->l = l;
    vk->w = w;
    pk->l = l;
    pk->w = w;

    for (unsigned long i = 0; i < l; ++i)
        mpz_init(vk->vi[i]);

    mpz_clear(t1);
    mpz_clear(t2);
    mpz_clear(t3);
    mpz_clear(t4);

    return 1;
}

void pcs_t_r_encrypt(pcs_t_public_key *pk, hcs_random *hr,
        mpz_t rop, mpz_t r, mpz_t plain1)
{
    mpz_t t1;
    mpz_init(t1);

    mpz_random_in_mult_group(r, hr->rstate, pk->n);
    mpz_powm(t1, pk->g, plain1, pk->n2);
    mpz_powm(rop, r, pk->n, pk->n2);
    mpz_mul(rop, rop, t1);
    mpz_mod(rop, rop, pk->n2);

    mpz_clear(t1);
}

void pcs_t_encrypt_r(pcs_t_public_key *pk, mpz_t rop, mpz_t r, mpz_t plain1)
{
    mpz_t t1;
    mpz_init(t1);

    mpz_powm(t1, pk->g, plain1, pk->n2);
    mpz_powm(rop, r, pk->n, pk->n2);
    mpz_mul(rop, rop, t1);
    mpz_mod(rop, rop, pk->n2);

    mpz_clear(t1);
}

void pcs_t_encrypt(pcs_t_public_key *pk, hcs_random *hr, mpz_t rop, mpz_t plain1)
{
    mpz_t t1;
    mpz_init(t1);

    mpz_random_in_mult_group(t1, hr->rstate, pk->n);
    mpz_powm(t1, t1, pk->n, pk->n2);
    mpz_powm(rop, pk->g, plain1, pk->n2);
    mpz_mul(rop, rop, t1);
    mpz_mod(rop, rop, pk->n2);

    mpz_clear(t1);
}

void pcs_t_reencrypt(pcs_t_public_key *pk, hcs_random *hr, mpz_t rop, mpz_t op)
{
    mpz_t t1;
    mpz_init(t1);

    mpz_random_in_mult_group(t1, hr->rstate, pk->n);
    mpz_powm(t1, t1, pk->n, pk->n2);
    mpz_mul(rop, op, t1);
    mpz_mod(rop, rop, pk->n2);

    mpz_clear(t1);
}

void pcs_t_ep_add(pcs_t_public_key *pk, mpz_t rop, mpz_t cipher1, mpz_t plain1)
{
    mpz_t t1;
    mpz_init(t1);

    mpz_set(t1, cipher1);
    mpz_powm(rop, pk->g, plain1, pk->n2);
    mpz_mul(rop, rop, t1);
    mpz_mod(rop, rop, pk->n2);

    mpz_clear(t1);
}

void pcs_t_ee_add(pcs_t_public_key *pk, mpz_t rop, mpz_t cipher1, mpz_t cipher2)
{
    mpz_mul(rop, cipher1, cipher2);
    mpz_mod(rop, rop, pk->n2);
}

void pcs_t_ep_mul(pcs_t_public_key *pk, mpz_t rop, mpz_t cipher1, mpz_t plain1)
{
    mpz_powm(rop, cipher1, plain1, pk->n2);
}

pcs_t_proof* pcs_t_init_proof(void)
{
    pcs_t_proof *pf = malloc(sizeof(pcs_t_proof));
    if (pf == NULL) return NULL;

    mpz_init(pf->e[0]);
    mpz_init(pf->e[1]);
    mpz_init(pf->a[0]);
    mpz_init(pf->a[1]);
    mpz_init(pf->z[0]);
    mpz_init(pf->z[1]);
    mpz_init(pf->generator);

    /* Default values */
    mpz_set_ui(pf->generator, 97);
    pf->m1 = 0;
    pf->m2 = 1;

    return pf;
}

void pcs_t_set_proof(pcs_t_proof *pf, mpz_t generator, unsigned long m1,
        unsigned long m2)
{
    mpz_set(pf->generator, generator);
    pf->m1 = m1;
    pf->m2 = m2;
}

void pcs_t_compute_ns_protocol(pcs_t_public_key *pk, hcs_random *hr,
        pcs_t_proof *pf, mpz_t cipher, mpz_t cipher_r, unsigned long id)
{
    mpz_t challenge, t1;
    mpz_init(challenge);
    mpz_init(t1);

    mpz_set(pf->e[0], cipher);

    // Random r in Zn* and a = E(0, r)
    mpz_set_ui(t1, 0);
    pcs_t_r_encrypt(pk, hr, t1, pf->a[0], t1);

    mpz_ripemd_mpz_ul(challenge, pf->a[0], id);

    mpz_powm(pf->z[0], cipher_r, challenge, pk->n2);
    mpz_mul(pf->z[0], pf->z[0], t1);
    mpz_mod(pf->z[0], pf->z[0], pk->n2);

    mpz_clear(t1);
    mpz_clear(challenge);
}

int pcs_t_verify_ns_protocol(pcs_t_public_key *pk, pcs_t_proof *pf,
        unsigned long id)
{
    int retval = 0;

    mpz_t t1, t2;
    mpz_init(t1);
    mpz_init(t2);

    /* Ensure u, a, z are prime to n */
    mpz_gcd(t1, pf->e[0], pk->n);
    if (mpz_cmp_ui(t1, 1) != 0)
        goto failure;

    mpz_gcd(t1, pf->a[0], pk->n);
    if (mpz_cmp_ui(t1, 1) != 0)
        goto failure;

    mpz_gcd(t1, pf->z[0], pk->n);
    if (mpz_cmp_ui(t1, 1) != 0)
        goto failure;

    mpz_set_ui(t1, 0);
    pcs_t_encrypt_r(pk, t1, pf->z[0], t1);
    mpz_ripemd_mpz_ul(pf->e[0], pf->a[0], id);
    mpz_powm(t2, pf->e[0], pf->e[0], pk->n2);
    mpz_mul(t2, t2, pf->a[0]);
    mpz_mod(t2, t2, pk->n2);

    if (mpz_cmp(t1, t2) != 0) {
        retval = 0;
    }

    retval = 1; /* Success */

failure:
    mpz_clear(t1);
    mpz_clear(t2);

    return retval;
}

void pcs_t_compute_1of2_ns_protocol(pcs_t_public_key *pk, hcs_random *hr,
        pcs_t_proof *pf, mpz_t cipher_m, mpz_t cipher_r, unsigned long nth_power, unsigned long id)
{
    mpz_t encrypt_value, other_value;
    mpz_init(encrypt_value);
    mpz_init(other_value);

    if (nth_power == pf->m1) {
        mpz_pow_ui(encrypt_value, pf->generator, pf->m1);
        mpz_pow_ui(other_value, pf->generator, pf->m2);
    }
    else if (nth_power == pf->m2) {
        mpz_pow_ui(encrypt_value, pf->generator, pf->m2);
        mpz_pow_ui(other_value, pf->generator, pf->m1);
    }
    else {
        return; /* Error */
    }

    /* Calculate proof */
    int choice = nth_power == pf->m1 ? 0 : 1;
    mpz_t r_hiding, t1, t2;
    mpz_init(r_hiding);
    mpz_init(t1);
    mpz_init(t2);

    mpz_random_in_mult_group(r_hiding, hr->rstate, pk->n2);
    mpz_powm(pf->a[choice], r_hiding, pk->n, pk->n2);

    mpz_random_in_mult_group(pf->z[1-choice], hr->rstate, pk->n2);
    mpz_urandomb(pf->e[1-choice], hr->rstate, HCS_HASH_SIZE);

    mpz_mul(t1, other_value, pk->n);
    mpz_mod(t1, t1, pk->n2);
    mpz_add_ui(t1, t1, 1);
    mpz_invert(t1, t1, pk->n2);
    mpz_mul(t1, t1, cipher_m);
    mpz_mod(t1, t1, pk->n2);
    mpz_powm(t1, t1, pf->e[1-choice], pk->n2);
    mpz_invert(t1, t1, pk->n2);
    mpz_powm(t2, pf->z[1-choice], pk->n, pk->n2);
    mpz_mul(t1, t1, t2);
    mpz_mod(pf->a[1-choice], t1, pk->n2);

    /* Construct a random challenge */
    mpz_t challenge;
    mpz_init(challenge);
    mpz_ripemd_3mpz_ul(challenge, pk->n, pf->a[0], pf->a[1], id);

    mpz_ui_pow_ui(t1, 2, HCS_HASH_SIZE);
    mpz_mod(t2, challenge, t1);
    mpz_sub(pf->e[choice], t2, pf->e[1-choice]);
    mpz_mod(pf->e[choice], pf->e[choice], t1);

    mpz_powm(t2, cipher_r, pf->e[choice], pk->n2);
    mpz_mul(t2, t2, r_hiding);
    mpz_mod(pf->z[choice], t2, pk->n2);

    mpz_clear(challenge);
    mpz_clear(r_hiding);
    mpz_clear(t1);
    mpz_clear(t2);
    mpz_clear(encrypt_value);
    mpz_clear(other_value);
}

int pcs_t_verify_1of2_ns_protocol(pcs_t_public_key *pk, pcs_t_proof *pf,
        mpz_t cipher, unsigned long id)
{
    int retval = 0;

    mpz_t pow2, esum, challenge, encrypt_value, t1, t2;
    mpz_init(pow2);
    mpz_init(challenge);
    mpz_init(esum);
    mpz_init(encrypt_value);
    mpz_init(t1);
    mpz_init(t2);

    mpz_ui_pow_ui(pow2, 2, HCS_HASH_SIZE);
    mpz_ripemd_3mpz_ul(challenge, pk->n, pf->a[0], pf->a[1], id);
    mpz_mod(challenge, challenge, pow2);
    mpz_add(esum, pf->e[0], pf->e[1]);
    mpz_mod(esum, esum, pow2);

    if (mpz_cmp(esum, challenge) != 0)
        goto failure;

    mpz_invert(encrypt_value, pk->g, pk->n2);
    mpz_mul(t1, cipher, encrypt_value);
    mpz_powm(t1, t1, pf->e[0], pk->n2);
    mpz_mul(t1, t1, pf->a[0]);
    mpz_mod(t1, t1, pk->n2);
    mpz_powm(t2, pf->z[0], pk->n, pk->n2);

    if (mpz_cmp(t1, t2) != 0)
        goto failure;

    mpz_mul(encrypt_value, pk->n, pf->generator);
    mpz_add_ui(encrypt_value, encrypt_value, 1);
    mpz_invert(encrypt_value, encrypt_value, pk->n2);
    mpz_mul(t1, cipher, encrypt_value);
    mpz_powm(t1, t1, pf->e[1], pk->n2);
    mpz_mul(t1, t1, pf->a[1]);
    mpz_mod(t1, t1, pk->n2);
    mpz_powm(t2, pf->z[1], pk->n, pk->n2);

    if (mpz_cmp(t1, t2) != 0)
        goto failure;

    mpz_init(pow2);
    mpz_init(challenge);
    mpz_init(esum);
    mpz_init(encrypt_value);
    mpz_init(t1);
    mpz_init(t2);

    retval = 1; /* Success */

failure:
    mpz_clear(pow2);
    mpz_clear(challenge);
    mpz_clear(esum);
    mpz_clear(encrypt_value);
    mpz_clear(t1);
    mpz_clear(t2);

    return retval;
}

void pcs_t_free_proof(pcs_t_proof *pf)
{
    mpz_clears(pf->e[0], pf->e[1], pf->a[0], pf->a[1], pf->z[0], pf->z[1], pf->generator, NULL);
    free(pf);
}

pcs_t_polynomial* pcs_t_init_polynomial(pcs_t_private_key *vk, hcs_random *hr)
{
    pcs_t_polynomial *px;

    if ((px = malloc(sizeof(pcs_t_polynomial))) == NULL)
        goto failure;
    if ((px->coeff = malloc(sizeof(mpz_t) * vk->w)) == NULL)
        goto failure;

    px->n = vk->w;
    mpz_init_set(px->coeff[0], vk->d);
    for (unsigned long i = 1; i < px->n; ++i) {
        mpz_init(px->coeff[i]);
        mpz_urandomm(px->coeff[i], hr->rstate, vk->nm);
    }

    return px;

failure:
    if (px->coeff) free(px->coeff);
    if (px) free(px);
    return NULL;
}

void pcs_t_compute_polynomial(pcs_t_private_key *vk, pcs_t_polynomial *px, mpz_t rop,
                              const unsigned long x)
{
    mpz_t t1, t2;
    mpz_init(t1);
    mpz_init(t2);

    mpz_set(rop, px->coeff[0]);
    for (unsigned long i = 1; i < px->n; ++i) {
        mpz_ui_pow_ui(t1, x + 1, i);        // Correct for server 0-indexing
        mpz_mul(t1, t1, px->coeff[i]);
        mpz_add(rop, rop, t1);
        mpz_mod(rop, rop, vk->nm);
    }

    mpz_clear(t1);
    mpz_clear(t2);
}

void pcs_t_free_polynomial(pcs_t_polynomial *px)
{
    for (unsigned long i = 0; i < px->n; ++i)
        mpz_clear(px->coeff[i]);
    free(px->coeff);
    free(px);
}

pcs_t_auth_server* pcs_t_init_auth_server(void)
{
    pcs_t_auth_server *au = malloc(sizeof(pcs_t_auth_server));
    if (!au) return NULL;

    mpz_init(au->si);
    return au;
}

void pcs_t_set_auth_server(pcs_t_auth_server *au, mpz_t si, unsigned long i)
{
    mpz_set(au->si, si);
    au->i = i + 1; // Input is assumed to be 0-indexed (from array)
}

/* Compute a servers share and set rop to the result. rop should usually
 * be part of an array so we can call pcs_t_share_combine with ease. */
void pcs_t_share_decrypt(pcs_t_public_key *pk, pcs_t_auth_server *au,
                         mpz_t rop, mpz_t cipher1)
{
    mpz_t t1;
    mpz_init(t1);

    mpz_mul(t1, au->si, pk->delta);
    mpz_mul_ui(t1, t1, 2);
    mpz_powm(rop, cipher1, t1, pk->n2);

    mpz_clear(t1);
}

/* c is expected to be of length vk->l, the number of servers. If the share
 * is not present, then it is expected to be equal to the value zero. */
int pcs_t_share_combine(pcs_t_public_key *pk, mpz_t rop, hcs_shares *hs)
{
    mpz_t t1, t2, t3;
    mpz_init(t1);
    mpz_init(t2);
    mpz_init(t3);

    mpz_set_ui(rop, 1);
    for (unsigned long i = 0; i < pk->l; ++i) {

        /* Skip zero shares */
        if (hs->flag[i] == 0)
            continue;

        /* Compute lagrange coefficients */
        mpz_set(t1, pk->delta);
        for (unsigned long j = 0; j < pk->l; ++j) {
            if ((j == i) || hs->flag[i] == 0)
                continue; /* i' in S\i and non-zero */

            long v = (long)j - (long)i;
            mpz_tdiv_q_ui(t1, t1, (v < 0 ? v*-1 : v));
            if (v < 0) mpz_neg(t1, t1);
            mpz_mul_ui(t1, t1, j + 1);
        }

        mpz_abs(t2, t1);
        mpz_mul_ui(t2, t2, 2);
        mpz_powm(t2, hs->shares[i], t2, pk->n2);

        if (mpz_sgn(t1) < 0 && !mpz_invert(t2, t2, pk->n2))
	        return 0;

        mpz_mul(rop, rop, t2);
        mpz_mod(rop, rop, pk->n2);
    }

    /* rop = c' */
    dlog_s(pk->n, rop, rop);
    mpz_pow_ui(t1, pk->delta, 2);
    mpz_mul_ui(t1, t1, 4);

    if (!mpz_invert(t1, t1, pk->n))
		return 0;

    mpz_mul(rop, rop, t1);
    mpz_mod(rop, rop, pk->n);

    mpz_clear(t1);
    mpz_clear(t2);
    mpz_clear(t3);
    return 1;
}

void pcs_t_free_auth_server(pcs_t_auth_server *au)
{
    mpz_clear(au->si);
    free(au);
}

void pcs_t_clear_public_key(pcs_t_public_key *pk)
{
    mpz_zeros(pk->g, pk->n, pk->n2, pk->delta, NULL);
}

void pcs_t_clear_private_key(pcs_t_private_key *vk)
{
    mpz_zeros(vk->v, vk->nm, vk->n,
              vk->n2, vk->d, NULL);

    if (vk->vi) {
        for (unsigned long i = 0; i < vk->l; ++i)
            mpz_clear(vk->vi[i]);
        free (vk->vi);
    }
}

void pcs_t_free_public_key(pcs_t_public_key *pk)
{
    mpz_clears(pk->g, pk->n, pk->n2, pk->delta, NULL);
    free(pk);
}

void pcs_t_free_private_key(pcs_t_private_key *vk)
{
    mpz_clears(vk->v, vk->nm, vk->n,
               vk->n2, vk->d, NULL);

    if (vk->vi) {
        for (unsigned long i = 0; i < vk->l; ++i)
            mpz_clear(vk->vi[i]);
        free (vk->vi);
    }

    free(vk);
}

int pcs_t_verify_key_pair(pcs_t_public_key *pk, pcs_t_private_key *vk)
{
    return mpz_cmp(vk->n, pk->n) == 0;
}

char *pcs_t_export_public_key(pcs_t_public_key *pk)
{
    char *buffer;
    char *retstr;

    JSON_Value *root = json_value_init_object();
    JSON_Object *obj  = json_value_get_object(root);
    buffer = mpz_get_str(NULL, HCS_INTERNAL_BASE, pk->n);
    json_object_set_string(obj, "n", buffer);
    json_object_set_number(obj, "w", pk->w);
    json_object_set_number(obj, "l", pk->l);
    retstr = json_serialize_to_string(root);

    json_value_free(root);
    free(buffer);
    return retstr;
}

char *pcs_t_export_proof(pcs_t_proof *pf)
{
    // Export all proof values
    char *buffer;
    char *retstr;

    JSON_Value *root = json_value_init_object();
    JSON_Object *obj = json_value_get_object(root);

#define sz__(x)  mpz_sizeinbase(pf->x, HCS_INTERNAL_BASE)

    const size_t buffer_len =
        HCS_MAX2(HCS_MAX3(sz__(e[0]), sz__(e[1]), sz__(a[0])),
            HCS_MAX3(sz__(a[1]), sz__(z[0]), sz__(z[1])));

#undef sz__

    buffer = malloc(buffer_len + 2);

    mpz_get_str(buffer, HCS_INTERNAL_BASE, pf->e[0]);
    json_object_set_string(obj, "e1", buffer);
    mpz_get_str(buffer, HCS_INTERNAL_BASE, pf->e[1]);
    json_object_set_string(obj, "e2", buffer);
    mpz_get_str(buffer, HCS_INTERNAL_BASE, pf->a[0]);
    json_object_set_string(obj, "a1", buffer);
    mpz_get_str(buffer, HCS_INTERNAL_BASE, pf->a[1]);
    json_object_set_string(obj, "a2", buffer);
    mpz_get_str(buffer, HCS_INTERNAL_BASE, pf->z[0]);
    json_object_set_string(obj, "z1", buffer);
    mpz_get_str(buffer, HCS_INTERNAL_BASE, pf->z[1]);
    json_object_set_string(obj, "z2", buffer);
    mpz_get_str(buffer, HCS_INTERNAL_BASE, pf->generator);
    json_object_set_string(obj, "generator", buffer);

    json_object_set_number(obj, "m1", pf->m1);
    json_object_set_number(obj, "m2", pf->m2);
    retstr = json_serialize_to_string(root);

    json_value_free(root);
    free(buffer);
    return retstr;
}

// TODO: IMPLEMENT
char *pcs_t_export_verify_values(pcs_t_private_key *vk)
{
    (void) vk;
    return "";
}

char *pcs_t_export_auth_server(pcs_t_auth_server *au)
{
    char *buffer;
    char *retstr;

    JSON_Value *root = json_value_init_object();
    JSON_Object *obj  = json_value_get_object(root);
    buffer = mpz_get_str(NULL, HCS_INTERNAL_BASE, au->si);
    json_object_set_string(obj, "si", buffer);
    json_object_set_number(obj, "i", au->i);
    retstr = json_serialize_to_string(root);

    json_value_free(root);
    free(buffer);
    return retstr;
}

int pcs_t_import_public_key(pcs_t_public_key *pk, const char *json)
{
    JSON_Value *root = json_parse_string(json);
    JSON_Object *obj = json_value_get_object(root);
    mpz_set_str(pk->n, json_object_get_string(obj, "n"), HCS_INTERNAL_BASE);
    pk->l = json_object_get_number(obj, "l");
    pk->w = json_object_get_number(obj, "w");
    json_value_free(root);

    /* Calculate remaining values */
    mpz_add_ui(pk->g, pk->n, 1);
    mpz_pow_ui(pk->n2, pk->n, 2);
    mpz_fac_ui(pk->delta, pk->l);
    return 0;
}

int pcs_t_import_proof(pcs_t_proof *pf, const char *json)
{
    JSON_Value *root = json_parse_string(json);
    JSON_Object *obj = json_value_get_object(root);
    mpz_set_str(pf->e[0], json_object_get_string(obj, "e1"), HCS_INTERNAL_BASE);
    mpz_set_str(pf->e[1], json_object_get_string(obj, "e1"), HCS_INTERNAL_BASE);
    mpz_set_str(pf->a[0], json_object_get_string(obj, "a1"), HCS_INTERNAL_BASE);
    mpz_set_str(pf->a[1], json_object_get_string(obj, "a2"), HCS_INTERNAL_BASE);
    mpz_set_str(pf->z[0], json_object_get_string(obj, "z1"), HCS_INTERNAL_BASE);
    mpz_set_str(pf->z[1], json_object_get_string(obj, "z2"), HCS_INTERNAL_BASE);
    mpz_set_str(pf->generator, json_object_get_string(obj, "generator"), HCS_INTERNAL_BASE);
    pf->m1 = json_object_get_number(obj, "m1");
    pf->m2 = json_object_get_number(obj, "m2");

    return 0;
}

// TODO: IMPLEMENT
int pcs_t_import_verify_values(pcs_t_private_key *vk, const char *json)
{
    (void) json;
    (void) vk;
    return 0;
}

int pcs_t_import_auth_server(pcs_t_auth_server *au, const char *json)
{
    JSON_Value *root = json_parse_string(json);
    JSON_Object *obj = json_value_get_object(root);
    mpz_set_str(au->si, json_object_get_string(obj, "si"), HCS_INTERNAL_BASE);
    au->i = json_object_get_number(obj, "i");
    json_value_free(root);

    /* Calculate remaining values */
    return 0;
}

#ifdef MAIN
int main(int argc, char **argv) {
    pcs_t_private_key *vk = pcs_t_init_private_key();
    pcs_t_public_key *pk = pcs_t_init_public_key();
    hcs_random *hr = hcs_init_random();

    pcs_t_generate_key_pair(pk, vk, hr, 128, 2, 4);

    mpz_t cipher, cipher_r, t1;
    mpz_init(cipher);
    mpz_init(cipher_r);
    mpz_init(t1);

    unsigned long id = 0x5341515;
    int pow = atoi(argv[1]);

    pcs_t_proof *pf = pcs_t_init_proof();

    mpz_pow_ui(t1, pf->generator, pow);
    pcs_t_r_encrypt(pk, hr, cipher, cipher_r, t1);

    pcs_t_compute_1of2_ns_protocol(pk, hr, pf, cipher, cipher_r, pow, id);

    if (pcs_t_verify_1of2_ns_protocol(pk, pf, cipher, id)) {
        gmp_printf("%Zd is an encryption of either 97^0 or 97^1\n", cipher);
    }
    else {
        gmp_printf("%Zd is not an encryption none of 97^0 or 97^1\n", cipher);
    }
}
#endif
