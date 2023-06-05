/**
 * @file djcs.h
 *
 * The Damgard-Jurik scheme is a scheme which provides homomorphic addition,
 * and limited multiplication on encrypted data. It is a generalisation of the
 * Paillier cryptoscheme, where the modulus n^2 is extended to n^(s+1) for
 * s > 0.
 */

#ifndef HCS_DJCS_H
#define HCS_DJCS_H

#include <gmp.h>
#include "hcs_random.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Public key for use in the Paillier system.
 */
typedef struct {
    unsigned long s; /**< Ciphertext space exponent */
    mpz_t *n;        /**< Modulus. Higher powers are precomputed. Len(n) = s */
    mpz_t g;         /**< Precomputation: n + 1 */
} djcs_public_key;

/**
 * Private key for use in the Paillier system.
 */
typedef struct {
    unsigned long s; /**< Ciphertext space exponent */
    mpz_t *n;        /**< Modulus; Higher powers are precomputed. Len(n) = s */
    mpz_t d;         /**< lcm(p, q) where n = p * q */
    mpz_t mu;        /**< g^d (mod n^(s+1)) */
} djcs_private_key;

/**
 * Initialise a djcs_public_key and return a pointer to the newly created
 * structure.
 *
 * @return A pointer to an initialised djcs_public_key, NULL on allocation
 *         failure
 */
djcs_public_key*  djcs_init_public_key(void);

/**
 * Initialise a djcs_private_key and return a pointer to the newly created
 * structure.
 *
 * @return A pointer to an initialised djcs_private_key, NULL on allocation
 *         failure
 */
djcs_private_key* djcs_init_private_key(void);

/**
 * Initialise a key pair with modulus size @p bits. It is required that @p pk
 * and @p vk are initialised before calling this function. @p pk and @p vk are
 * expected to not be NULL.
 *
 * In practice the @p bits value should usually be greater than 2048 to ensure
 * sufficient security.
 *
 * Whilst @p s is variable and and can be any value upto a minimum of 32 bits,
 * beware when calling the function with large values. The modulus grows
 * exponentially given @p s, so even small values greater than 7 will incur
 * massive performance penalties, and increase the risk of overflow
 * (~40 million digits for the modulus for this to occur).
 *
 * \warning This function attempts to allocate memory, so calling this twice in
 * sucession with the same keys will cause your program to lose a pointer to
 * this allocated memory, resulting in a memory leak. If you wish to call this
 * function in this manner, ensure djcs_clear_public_key and/or
 * djcs_clear_private_key are called prior.
 *
 * @param pk A pointer to an initialised djcs_public_key
 * @param vk A pointer to an initialised djcs_private_key
 * @param hr A pointer to an initialised hcs_random type
 * @param s The size exponent for the ciphertext space we wish to work in
 * @param bits The number of bits for the modulus of the key
 */
int djcs_generate_key_pair(djcs_public_key *pk, djcs_private_key *vk,
                           hcs_random *hr, unsigned long s, unsigned long bits);

/**
 * Encrypt a value @p plain1, and set @p rop to the encrypted result.
 *
 * @param pk A pointer to an initialised djcs_public_key
 * @param hr A pointer to an initialised hcs_random type
 * @param rop mpz_t where the encrypted result is stored
 * @param plain1 mpz_t to be encrypted
 */
void djcs_encrypt(djcs_public_key *pk, hcs_random *hr, mpz_t rop, mpz_t plain1);

/**
 * Reencrypt an encrypted value @p op. Upon decryption, this newly
 * encrypted value, @p rop, will retain the same value as @p op.
 *
 * @param pk A pointer to an initialised djcs_public_key
 * @param hr A pointer to an initialised hcs_random type
 * @param rop mpz_t where the newly encrypted value is stored
 * @param op mpz_t to be reencrypted
 */
void djcs_reencrypt(djcs_public_key *pk, hcs_random *hr, mpz_t rop, mpz_t op);

/**
 * Add a plaintext value @p plain1 to an encrypted value @p cipher1, storing
 * the result in @p rop.
 *
 * @param pk A pointer to an initialised djcs_public_key
 * @param rop mpz_t where the newly encrypted value is stored
 * @param cipher1 mpz_t to be added together
 * @param plain1 mpz_t to be added together
 */
void djcs_ep_add(djcs_public_key *pk, mpz_t rop, mpz_t cipher1, mpz_t plain1);

/**
 * Add an encrypted value @p cipher2 to an encrypted value @p cipher1, storing
 * the result in @p rop.
 *
 * @param pk A pointer to an initialised djcs_public_key.
 * @param rop mpz_t where the newly encrypted value is stored
 * @param cipher1 mpz_t to be added together
 * @param cipher2 mpz_t to be added together
 */
void djcs_ee_add(djcs_public_key *pk, mpz_t rop, mpz_t cipher1, mpz_t cipher2);

/**
 * Multiply a plaintext value @p plain1 with an encrypted value @p cipher1,
 * storing the result in @p rop.
 *
 * @param pk A pointer to an initialised djcs_public_key
 * @param rop Where the new encrypted result is stored
 * @param cipher1 The encrypted value which is to be multipled to
 * @param plain1 The plaintext value which is to be multipled
 */
void djcs_ep_mul(djcs_public_key *pk, mpz_t rop, mpz_t cipher1, mpz_t plain1);

/**
 * Decrypt a value @p cipher1, and set @p rop to the decrypted result. @p rop
 * and @p cipher1 can aliases for the same mpz_t.
 *
 * @param vk A pointer to an initialised djcs_private_key
 * @param rop mpz_t where the decrypted result is stored
 * @param cipher1 mpz_t to be decrypted
 */
void djcs_decrypt(djcs_private_key *vk, mpz_t rop, mpz_t cipher1);

/**
 * Clears all data in a djcs_public_key. This does not free memory in the
 * keys, only putting it into a state whereby they can be safely used to
 * generate new key values.
 *
 * @param pk A pointer to an initialised djcs_public_key
 */
void djcs_clear_public_key(djcs_public_key *pk);

/**
 * Clears all data in a djcs_private_key. This does not free memory in the
 * keys, only putting it into a state whereby they can be safely used to
 * generate new key values.
 *
 * @param vk A pointer to an initialised djcs_private_key
 */
void djcs_clear_private_key(djcs_private_key *vk);

/**
 * Frees a djcs_public_key and all associated memory. The key memory is
 * not zeroed, so one must call djcs_clear_public_key if it is required.
 * one does not need to call djcs_clear_public_key before using this function.
 *
 * @param pk A pointer to an initialised djcs_public_key
 */
void djcs_free_public_key(djcs_public_key *pk);

/**
 * Frees a djcs_private_key and all associated memory. The key memory is
 * not zeroed, so one must call djcs_clear_private_key if it is required.
 * one does not need to call djcs_clear_private_key before using this function.
 *
 * @param vk v pointer to an initialised djcs_private_key
 */
void djcs_free_private_key(djcs_private_key *vk);

/**
 * Check certain values shared between public and private keys to ensure
 * they indeed are pairs. This checks only the n and s values, and assumes that
 * the caller has not altered other internal values. If the caller has only
 * interacted with the keys through the usual functions, then this should
 * guarantee the keys are pairs.
 *
 * @param pk A pointer to an initialised djcs_public_key
 * @param vk A pointer to an initialised djcs_private_key
 * @return non-zero if keys are valid, else zero
 */
int djcs_verify_key_pair(djcs_public_key *pk, djcs_private_key *vk);

/**
 * Export a public key as a string. We only store the minimum required values
 * to restore the key. In this case, these are the s and n values.
 *
 * The format these strings export as is as a JSON object.
 *
 * @param pk A pointer to an initialised djcs_public_key
 * @return A string representing the given key, else NULL on error
 */
char* djcs_export_public_key(djcs_public_key *pk);

/**
 * Export a private key as a string. We only store the minimum required values
 * to restore the key. In this case, these are the s, n and d values. The
 * remaining values are then computed from these on import.
 *
 * @param vk A pointer to an initialised djcs_private_key
 * @return A string representing the given key, else NULL on error
 */
char* djcs_export_private_key(djcs_private_key *vk);

/**
 * Import a public key from a string. The input string is expected to
 * match the format given by the export functions.
 *
 * @param pk A pointer to an initialised djcs_public_key
 * @param json A string storing the contents of a public key
 * @return non-zero if success, else zero on format error
 */
int djcs_import_public_key(djcs_public_key *pk, const char *json);

/**
 * Import a private key from a string. The input string is expected to
 * match the format given by the export functions.
 *
 * @param vk A pointer to an initialised djcs_private_key
 * @param json A string storing the contents of a private key
 * @return non-zero if success, else zero on format error
 */
int djcs_import_private_key(djcs_private_key *vk, const char *json);

#ifdef __cplusplus
}
#endif

#endif
