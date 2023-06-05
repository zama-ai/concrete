/**
 * @file djcs_t.h
 *
 * The threshold Damgard-Jurik scheme is a generalization of the Paillier
 * cryptoscheme, but with an increased ciphertext space.
 *
 * All mpz_t values can be aliases unless otherwise stated.
 *
 * \warning All indexing for the servers and polynomial functions should be
 * zero-indexed, as is usual when working with c arrays. The functions
 * themselves correct for this internally, and 1-indexing servers may result
 * in incorrect results.
 */

#ifndef HCS_DJCS_T_H
#define HCS_DJCS_T_H

#include <gmp.h>
#include "hcs_random.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Details that a decryption server is required to keep track of.
 */
typedef struct {
    unsigned long i;    /**< The server index of this particular instance */
    mpz_t si;           /**< The polynomial evaluation at @p i */
} djcs_t_auth_server;

/**
 * Public key for use in the Threshold Damgard-Jurik system.
 */
typedef struct {
    unsigned long s; /**< Ciphertext space exponent */
    mpz_t *n;        /**< Modulus of the key. n = p * q */
    mpz_t g;         /**< Precomputation: n + 1 usually, may be 2 */
} djcs_t_public_key;

/**
 * Private key for use in the Threshold Damgard-Jurik system.
 */
typedef struct {
    unsigned long s;    /**< Ciphertext space exponent */
    unsigned long w;    /**< The number of servers req to decrypt */
    unsigned long l;    /**< The number of decryption servers */
    mpz_t *vi;          /**< Verification values for the decrypt servers */
    mpz_t *n;           /**< Modulus; Higher powers are precomputed. Len(n) = s*/
    mpz_t v;            /**< Cyclic generator of squares in Z*n^2 */
    mpz_t delta;        /**< Precomputation: l! */
    mpz_t d;            /**< d = 0 mod m and d = 1 mod n^2 */
    mpz_t p;            /**< A random prime determined during key generation */
    mpz_t ph;           /**< A random prime such that p = 2*ph + 1 */
    mpz_t q;            /**< A random prime determined during key generation */
    mpz_t qh;           /**< A random prime such that q = 2*qh + 1 */
    mpz_t m;            /**< Precomputation: ph * qh */
    mpz_t nsm;          /**< Precomputation: n * m */
} djcs_t_private_key;

/**
 * Initialise a djcs_t_public_key and return a pointer to the newly created
 * structure.
 *
 * @return A pointer to an initialised djcs_t_public_key, NULL on allocation
 *         failure
 */
djcs_t_public_key* djcs_t_init_public_key(void);

/**
 * Initialise a djcs_private_key and return a pointer to the newly created
 * structure.
 *
 * @return A pointer to an initialise djcs_t_private_key, NULL on allocation
 *         failure
 */
djcs_t_private_key* djcs_t_init_private_key(void);

/**
 * Initialise a key pair with modulus size @p bits. It is required that @p pk
 * and @p vk are initialised before calling this function. @p pk and @p vk are
 * expected to not be NULL.
 *
 * In practice the @p bits value should usually be greater than 2048 to ensure
 * sufficient security.
 *
 * Whilst @p s is variable and can be any value upto a minimum of 32 bits,
 * beware when calling the function with large values. The modulus grows
 * exponentially given @p s, so even small values greater than 7 will incur
 * massive performance penalties, and increase the risk of overflow
 * (~40 million digits for the modulus for this to occur).
 *
 * \warning This function attempts to allocate memory, so calling this twice
 * in succession with the same keys will cause your program to lose a pointer
 * to this allocated memory, resulting in a memory leak. If you wish to call
 * this function in this manner, ensure djcs_t_clear_public_key and/or
 * djcs_t_clear_private_key are called prior.
 */
void djcs_t_generate_key_pair(djcs_t_public_key *pk, djcs_t_private_key *vk,
        hcs_random *hr, const unsigned long s, const unsigned long bits,
        const unsigned long l, const unsigned long w);

/**
 * Encrypt a value @p plain1, and set @p rop to the encrypted result.
 *
 * @param pk A pointer to an initialised djcs_t_public_key
 * @param hr A pointer to an initialised hcs_random
 * @param rop mpz_t where the encrypted result is stored
 * @param plain1 mpz_t to be encrypted
 */
void djcs_t_encrypt(djcs_t_public_key *pk, hcs_random *hr, mpz_t rop,
                    mpz_t plain1);

/**
 * Reencrypt an encrypted value @p op. Upon decryption, this newly
 * encrypted value, @p rop, will retain the same value as @p op.
 *
 * @param pk A pointer to an initialised djcs_t_public_key
 * @param hr A pointer to an initialised hcs_random
 * @param rop mpz_t where the newly encrypted value is stored
 * @param op mpz_t to be reencrypted
 */
void djcs_t_reencrypt(djcs_t_public_key *pk, hcs_random *hr, mpz_t rop, mpz_t op);

/**
 * Add a plaintext value @p plain1 to an encrypted value @p cipher1, storing
 * the result in @p rop.
 *
 * @param pk A pointer to an initialised djcs_t_public_key
 * @param rop mpz_t where the newly encrypted value is stored
 * @param cipher1 mpz_t to be added together
 * @param plain1 mpz_t to be added together
 */
void djcs_t_ep_add(djcs_t_public_key *pk, mpz_t rop, mpz_t cipher1, mpz_t plain1);

/**
 * Add an encrypted value @p cipher2 to an encrypted value @p cipher1, storing
 * the result in @p rop.
 *
 * @param pk A pointer to an initialised djcs_t_public_key.
 * @param rop mpz_t where the newly encrypted value is stored
 * @param cipher1 mpz_t to be added together
 * @param cipher2 mpz_t to be added together
 */
void djcs_t_ee_add(djcs_t_public_key *pk, mpz_t rop, mpz_t cipher1, mpz_t cipher2);

/**
 * Multiply a plaintext value @p plain1 with an encrypted value @p cipher1,
 * storing the result in @p rop.
 *
 * @param pk A pointer to an initialised djcs_t_public_key
 * @param rop Where the new encrypted result is stored
 * @param cipher1 The encrypted value which is to be multipled to
 * @param plain1 The plaintext value which is to be multipled
 */
void djcs_t_ep_mul(djcs_t_public_key *pk, mpz_t rop, mpz_t cipher1, mpz_t plain1);

/**
 * Allocate and initialise the values in a random polynomial. The length of
 * this polynomial is taken from values in @p vk, specifically it will be
 * of length vk->w. The polynomial functions are to be used by a single trusted
 * party, for which once the required computation is completed, the polynomial
 * can be discarded.
 *
 * @param vk v pointer to an initialised djcs_t_private_key
 * @param hr A pointer to an initialised hcs_random type
 * @return A polynomial coefficient list on success, else NULL
 */
mpz_t* djcs_t_init_polynomial(djcs_t_private_key *vk, hcs_random *hr);

/**
 * Compute a polynomial P(x) for a given x value in the required finite field.
 * The coefficients should be given as a list of mpz_t values, computed via the
 * djcs_t_init_polynomial function.
 *
 * @param vk A pointer to an initialised djcs_t_private_key
 * @param coeff A pointer to a list of coefficients of a polynomial
 * @param rop mpz_t where the result is stored
 * @param x The value to calculate the polynomial at
 */
void djcs_t_compute_polynomial(djcs_t_private_key *vk, mpz_t *coeff, mpz_t rop,
                               const unsigned long x);

/**
 * Frees a given polynomial (array of mpz_t values) and all associated data.
 * The same private key which was used to generate these values should be used
 * as an argument.
 *
 * @param vk A pointer to an initialised djcs_t_private_key
 * @param coeff A pointer to a list of coefficients of a polynomial
 */
void djcs_t_free_polynomial(djcs_t_private_key *vk, mpz_t *coeff);

/**
 * Initialise a djcs_t_auth_server and return a pointer to the newly created
 * structure.
 *
 * @return A pointer to an initialised djcs_t_auth_server, NULL on allocation
 *         failure
 */
djcs_t_auth_server* djcs_t_init_auth_server(void);

/**
 * Set the internal values for the server @p au. @p si is the secret polynomial
 * share for the given value, @p i. These values should be shared in a secret
 * and secure way and not given out publicly. The index given to each server
 * should be unique.
 *
 * @param au A pointer to an initialised djcs_t_auth_server
 * @param si The value of a secret polynomial evaluated at @p i
 * @param i The servers given index
 */
void djcs_t_set_auth_server(djcs_t_auth_server *au, mpz_t si, unsigned long i);

/**
 * For a given ciphertext @p cipher1, compute the server @p au's share and store
 * the result in the variable @p rop. These shares can be managed, and then
 * combined when sufficient shares have been accumulated using the
 * djcs_t_share_combine function.
 *
 * @param vk A pointer to an initialised djcs_t_private_key
 * @param au A pointer to an initialised djcs_t_auth_server
 * @param rop mpz_t where the calculated share is stored
 * @param cipher1 mpz_t which stores the ciphertext to decrypt
 */
void djcs_t_share_decrypt(djcs_t_private_key *vk, djcs_t_auth_server *au,
                          mpz_t rop, mpz_t cipher1);

/**
 * Combine an array of shares @p c, storing the result in @p rop.
 *
 * The array @p c must be managed by the caller, and is expected to be at least
 * of length vk->l. If it is greater, only the first vk->l values are scanned.
 * If a share is not present, then the value is expected to be 0. If reusing the
 * same array for a number of decryptions, ensure that the array is zeroed
 * between each combination.
 *
 * \todo Potentially construct a proper type for storing a list of shares to
 *       ensure these functions are called in the correct way.
 *
 * \pre vk->l <= length(c)
 *
 * @param vk A pointer to an initialised djcs_t_private_key
 * @param rop mpz_t where the combined decrypted result is stored
 * @param c array of share values
 */
void djcs_t_share_combine(djcs_t_private_key *vk, mpz_t rop, mpz_t *c);

/**
 * Frees a djcs_t_auth_server and all associated memory.
 *
 * @param au A pointer to an initialised djcs_t_auth_server
 */
void djcs_t_free_auth_server(djcs_t_auth_server *au);

/**
 * Clears all data in a djcs_t_public_key. This does not free memory in the
 * keys, only putting it into a state whereby they can be safely used to
 * generate new key values.
 *
 * @param pk A pointer to an initialised djcs_t_public_key
 */
void djcs_t_clear_public_key(djcs_t_public_key *pk);

/**
 * Clears all data in a djcs_t_private_key. This does not free memory in the
 * keys, only putting it into a state whereby they can be safely used to
 * generate new key values.
 *
 * @param vk A pointer to an initialised djcs_t_private_key
 */
void djcs_t_clear_private_key(djcs_t_private_key *vk);

/**
 * Frees a djcs_t_public_key and all associated memory. The key memory is
 * not zeroed, so one must call djcs_t_clear_public_key if it is required.
 * one does not need to call djcs_t_clear_public_key before using this function.
 *
 * @param pk A pointer to an initialised djcs_t_public_key
 */
void djcs_t_free_public_key(djcs_t_public_key *pk);

/**
 * Frees a djcs_t_private_key and all associated memory. The key memory is
 * not zeroed, so one must call djcs_t_clear_private_key if it is required.
 * one does not need to call djcs_t_clear_private_key before using this function.
 *
 * @param vk v pointer to an initialised djcs_t_private_key
 */
void djcs_t_free_private_key(djcs_t_private_key *vk);

#ifdef __cplusplus
}
#endif

#endif
