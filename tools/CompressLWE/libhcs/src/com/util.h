/**
 * @file util.h
 *
 * Internal common functions that are not exposed to users of this library.
 */

#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>

#ifdef __cplusplus
extern "C" {
#endif

// Error codes
#define HCS_OK    0x0
#define HCS_EOPEN 0x1
#define HCS_EREAD 0x2

#define HCS_INTERNAL_BASE 62

#define HCS_MAX2(x,y) ((x) > (y) ? (x) : (y))
#define HCS_MAX3(x,y,z) HCS_MAX2(HCS_MAX2(x, y), z)
#define HCS_MAX4(w,x,y,z) HCS_MAX2(HCS_MAX3(w, x, y), z)

/**
 * Zeroes a all memory allocated to a mpz_t @p op.
 */
void mpz_zero(mpz_t op);

/**
 * Zeroes a va_list of mpz_t.
 */
void mpz_zeros(mpz_t op, ...);

/* These are generally not called directly, except for testing purposes */
void internal_fast_random_prime(mpz_t rop, gmp_randstate_t rstate, mp_bitcnt_t bitcnt);
void internal_naive_random_prime(mpz_t rop, gmp_randstate_t rstate, mp_bitcnt_t bitcnt);
void internal_naive_random_safe_prime(mpz_t rop1, mpz_t rop2, gmp_randstate_t rstate,
        mp_bitcnt_t bitcnt);
void internal_fast_random_safe_prime(mpz_t rop1, mpz_t rop2, gmp_randstate_t rstate,
        mp_bitcnt_t bitcnt);

/**
 * Generate a random prime @p rop of bit length > @p bitcnt
 */
#define mpz_random_prime(rop1, rstate, bitcnt) \
    internal_naive_random_prime(rop1, rstate, bitcnt)

/**
 * Generate a random safe prime @p rop of bit length > @p bitcnt.
 */
#define mpz_random_safe_prime(rop1, rop2, rstate, bitcnt) \
    internal_naive_random_safe_prime(rop1, rop2, rstate, bitcnt)

/**
 * Gather seed from OS' entropy pool and store in an mpz_t @p rop. All values
 * can be bitwise xored together.
 *
 * @return
 * HCS_OK    - Success
 * HCS_EOPEN - Failed to open entropy source
 * HCS_EREAD - Failued to read from entropy source
 */
int mpz_seed(mpz_t rop, int bits);

/**
 * Perform the chinese remainder theorem on 2 congruences.
 */
void mpz_2crt(mpz_t rop, mpz_t con1_a, mpz_t con1_m, mpz_t con2_a,
              mpz_t con2_m);

/**
 * Generate a random dsa prime @p rop.
 */
void mpz_random_dsa_prime(mpz_t rop, gmp_randstate_t rstate,
                          mp_bitcnt_t bitcnt);

/**
 * Generate a random value in the multiplicative group @p op*, storing the
 * result in @p rop.
 */
void mpz_random_in_mult_group(mpz_t rop, gmp_randstate_t rstate, mpz_t op);

void mpz_ripemd_mpz_ul(mpz_t rop, mpz_t op1, unsigned long op2);
void mpz_ripemd_3mpz_ul(mpz_t rop, mpz_t op1, mpz_t op2, mpz_t op3, unsigned long op4);

#ifdef __cplusplus
}
#endif

#endif
