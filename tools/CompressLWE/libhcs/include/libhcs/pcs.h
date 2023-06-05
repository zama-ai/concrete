/**
 * @file pcs.h
 *
 * The Paillier scheme is a scheme which provides homormorphic addition, and
 * limited multiplication on encrypted data. These can be summarised as:
 *
 * @code
 * E(a + b) = pcs_ee_add(E(a), E(b));
 * E(a + b) = pcs_ep_add(E(a), b);
 * E(a * b) = pcs_ep_mul(E(a), b);
 * @endcode
 *
 * All mpz_t values can be aliases unless otherwise stated.
 */

#ifndef HCS_PCS_H
#define HCS_PCS_H

#include <gmp.h>
#include "hcs_random.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Public key for use in the Paillier system.
 */
typedef struct {
    mpz_t n;        /**< Modulus of the key: n = p * q */
    mpz_t g;        /**< Precomputation: n + 1 usually, be 2*/
    mpz_t n2;       /**< Precomputation: n^2 */
} pcs_public_key;

/**
 * Private key for use in the Paillier system.
 */
typedef struct {
    mpz_t p;        /**< A random prime determined during key generation */
    mpz_t q;        /**< A random prime determined during key generation */
    mpz_t p2;       /**< Precomputation: p^2 */
    mpz_t q2;       /**< Precomputation: q^2 */
    mpz_t hp;       /**< Precomputation: L_p(g^{p-1} mod p^2)^{-1} mod p */
    mpz_t hq;       /**< Precomputation: L_p(g^{q-1} mod q^2)^{-1} mod q */
    mpz_t lambda;   /**< Precomputation: euler-phi(p, q) */
    mpz_t mu;       /**< Precomputation: lambda^{-1} mod n */
    mpz_t n;        /**< Precomputation: p * q */
    mpz_t n2;       /**< Precomputation: n^2 */
} pcs_private_key;

/**
 * Initialise a pcs_public_key and return a pointer to the newly created
 * structure.
 *
 * @return A pointer to an initialised pcs_public_key, NULL on allocation
 *         failure
 */
pcs_public_key*  pcs_init_public_key(void);

/**
 * Initialise a pcs_private_key and return a pointer to the newly created
 * structure.
 *
 * @return A pointer to an initialised pcs_private_key, NULL on allocation
 *         failure
 */
pcs_private_key* pcs_init_private_key(void);

/**
 * Initialise a key pair with modulus size @p bits. It is required that @p pk
 * and @p vk are initialised before calling this function. @p pk and @p vk are
 * expected to not be NULL.
 *
 * In practice the @p bits value should usually be greater than 2048 to ensure
 * sufficient security.
 *
 * @code
 * pcs_public_key *pk = pcs_init_public_key();
 * pcs_private_key *vk = pcs_init_private_key();
 * hcs_random = hcs_random_init();
 * pcs_generate_key(pk, vk, hr, 2048);
 * @endcode
 *
 * @param pk A pointer to an initialised pcs_public_key
 * @param vk A pointer to an initialised pcs_private_key
 * @param hr A pointer to an initialised hcs_random type
 * @param bits The number of bits for the modulus of the key
 */
void pcs_generate_key_pair(pcs_public_key *pk, pcs_private_key *vk,
                           hcs_random *hr, const unsigned long bits);

/**
 * Encrypt a value @p plain1, and set @p rop to the encrypted result.
 *
 * @param pk A pointer to an initialised pcs_public_key
 * @param hr A pointer to an initialised hcs_random type
 * @param rop mpz_t where the encrypted result is stored
 * @param plain1 mpz_t to be encrypted
 */
void pcs_encrypt(pcs_public_key *pk, hcs_random *hr, mpz_t rop, mpz_t plain1);

/**
 * Encrypt a value @p plain1, and set @p rop to the encrypted result. Do not
 * randomly generate an r value, instead, use the given @p r. This is largely
 * useless to a user, but is important for some zero-knowledge proofs.
 *
 * @param pk A pointer to an initialised pcs_public_key
 * @param rop mpz_t where the encrypted result is stored
 * @param plain1 mpz_t to be encrypted
 * @param r random mpz_t value to be used during encryption
 */
void pcs_encrypt_r(pcs_public_key *pk, mpz_t rop, mpz_t plain1, mpz_t r);

/**
 * Reencrypt an encrypted value @p op. Upon decryption, this newly
 * encrypted value, @p rop, will retain the same value as @p op.
 *
 * @param pk A pointer to an initialised pcs_public_key
 * @param hr A pointer to an initialised hcs_random type
 * @param rop mpz_t where the newly encrypted value is stored
 * @param op mpz_t to be reencrypted
 */
void pcs_reencrypt(pcs_public_key *pk, hcs_random *hr, mpz_t rop, mpz_t op);

/**
 * Add a plaintext value @p plain1 to an encrypted value @p cipher1, storing
 * the result in @p rop.
 *
 * @param pk A pointer to an initialised pcs_public_key
 * @param rop mpz_t where the newly encrypted value is stored
 * @param cipher1 mpz_t to be added together
 * @param plain1 mpz_t to be added together
 */
void pcs_ep_add(pcs_public_key *pk, mpz_t rop, mpz_t cipher1, mpz_t plain1);

/**
 * Add an encrypted value @p cipher2 to an encrypted value @p cipher1, storing
 * the result in @p rop.
 *
 * @param pk A pointer to an initialised pcs_public_key.
 * @param rop mpz_t where the newly encrypted value is stored
 * @param cipher1 mpz_t to be added together
 * @param cipher2 mpz_t to be added together
 */
void pcs_ee_add(pcs_public_key *pk, mpz_t rop, mpz_t cipher1, mpz_t cipher2);

/**
 * Multiply a plaintext value @p plain1 with an encrypted value @p cipher1,
 * storing the result in @p rop. All the parameters can be aliased, however,
 * usually only @p rop and @p cipher1 will be.
 *
 * @param pk A pointer to an initialised pcs_public_key.
 * @param rop mpz_t where the newly encrypted value is stored
 * @param cipher1 mpz_t to be multiplied together
 * @param plain1 mpz_t to be multiplied together
 */
void pcs_ep_mul(pcs_public_key *pk, mpz_t rop, mpz_t cipher1, mpz_t plain1);

/**
 * Decrypt a value @p cipher1, and set @p rop to the decrypted result. @p rop
 * and @p cipher1 can aliases for the same mpz_t.
 *
 * @param vk A pointer to an initialised pcs_private_key
 * @param rop mpz_t where the decrypted result is stored
 * @param cipher1 mpz_t to be decrypted
 */
void pcs_decrypt(pcs_private_key *vk, mpz_t rop, mpz_t cipher1);

/**
 * This function zeros all data in @p pk. It is useful to use if we wish
 * to generate or import a new value for the given pcs_public_key and want
 * to safely ensure the old values are removed.
 *
 * @code
 * // ... Initialised a key pk and done some work with it
 *
 * pcs_clear_public_key(pk); // All data from old key is now gone
 * pcs_import_public_key(pk, "public.key"); // Safe to reuse this key
 * @endcode
 *
 * @param pk A pointer to an initialised pcs_public_key
 */
void pcs_clear_public_key(pcs_public_key *pk);

/**
 * This function zeros all data in @p pk. It is useful to use if we wish
 * to generate or import a new value for the given pcs_private_key and want
 * to safely ensure the old values are removed.
 *
 * @param vk A pointer to an initialised pcs_private_key
 */
void pcs_clear_private_key(pcs_private_key *vk);

/**
 * Frees a pcs_public_key and all associated memory. The key memory is
 * not zeroed, so one must call pcs_clear_public_key if it is required.
 * one does not need to call pcs_clear_public_key before using this function.
 *
 * @param pk A pointer to an initialised pcs_public_key
 */
void pcs_free_public_key(pcs_public_key *pk);

/**
 * Frees a pcs_private_key and all associated memory. The key memory is
 * not zeroed, so one must call pcs_clear_private_key if it is required.
 * one does not need to call pcs_clear_private_key before using this function.
 *
 * @param vk v pointer to an initialised pcs_private_key
 */
void pcs_free_private_key(pcs_private_key *vk);

/**
 * Check certain values shared between public and private keys to ensure
 * they indeed are pairs. This checks only the n values, and assumes that
 * the caller has not altered other internal values. If the caller has only
 * interacted with the keys through the usual functions, then this should
 * guarantee the keys are pairs.
 *
 * @param pk A pointer to an initialised pcs_public_key
 * @param vk A pointer to an initialised pcs_private_key
 * @return non-zero if keys are valid, else zero
 */
int pcs_verify_key_pair(pcs_public_key *pk, pcs_private_key *vk);

/**
 * Export a public key as a string. We only store the minimum required values
 * to restore the key. In this case, this is only the n value.
 *
 * The format these strings export as is as a JSON object.
 *
 * @param pk A pointer to an initialised pcs_public_key
 * @return A string representing the given key, else NULL on error
 */
char* pcs_export_public_key(pcs_public_key *pk);

/**
 * Export a private key as a string. We only store the minimum required values
 * to restore the key. In this case, these are the p and q values. The
 * remaining values are then computed from these on import.
 *
 * @param vk A pointer to an initialised pcs_private_key
 * @return A string representing the given key, else NULL on error
 */
char* pcs_export_private_key(pcs_private_key *vk);

/**
 * Import a public key from a string. The input string is expected to
 * match the format given by the export functions.
 *
 * @param pk A pointer to an initialised pcs_public_key
 * @param json A string storing the contents of a public key
 * @return non-zero if success, else zero on format error
 */
int pcs_import_public_key(pcs_public_key *pk, const char *json);

/**
 * Import a private key from a string. The input string is expected to
 * match the format given by the export functions.
 *
 * @param vk A pointer to an initialised pcs_private_key
 * @param json A string storing the contents of a private key
 * @return non-zero if success, else zero on format error
 */
int pcs_import_private_key(pcs_private_key *vk, const char *json);

#ifdef __cplusplus
}
#endif

#endif
