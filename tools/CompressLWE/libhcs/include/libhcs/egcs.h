/**
 * @file egcs.h
 *
 * The ElGamal scheme is a scheme which provides homormophic multiplication.
 * It is usually used with some padding scheme similar to RSA. This
 * implementation only provides unpadded values to exploit this property.
 *
 * Due to the nature of the ciphertext, this implementation provides a
 * unique type for the ciphertext, rather than just using gmp mpz_t types.
 * These can be used with the provided initialisation functions, and should
 * be freed on program termination.
 *
 * All mpz_t values can be alises unless otherwise stated.
 */

#ifndef HCS_EGCS_H
#define HCS_EGCS_H

#include <gmp.h>
#include "hcs_random.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Ciphertext type for use in the ElGamal scheme.
 */
typedef struct {
    mpz_t c1;   /**< First value of egcs cipher */
    mpz_t c2;   /**< Second value of egcs cipher */
} egcs_cipher;

/**
 * Public key for use in the ElGamal scheme.
 */
typedef struct {
    mpz_t g;    /**< Generator for the cyclic group */
    mpz_t q;    /**< Order of the cyclic group */
    mpz_t h;    /**< g^x in G */
} egcs_public_key;

/**
 * Private key for use in the ElGamal scheme.
 */
typedef struct {
    mpz_t x;    /**< Random value in {1, ..., q-1} */
    mpz_t q;    /**< Order of the cyclic group */
} egcs_private_key;

/**
 * Initialise a egcs_public_key and return a pointer to the newly created
 * structure.
 *
 * @return A pointer to an initialised egcs_public_key, NULL on allocation
 *         failure
 */
egcs_public_key*  egcs_init_public_key(void);

/**
 * Initialise a egcs_private_key and return a pointer to the newly created
 * structure.
 *
 * @return A pointer to an initialised egcs_private_key, NULL on allocation
 *         failure
 */
egcs_private_key* egcs_init_private_key(void);

/**
 * Initialise a key pair with modulus size @p bits. It is required that @p pk
 * and @p vk are initialised before calling this function. @p pk and @p vk are
 * expected to not be NULL.
 *
 * In practice the @p bits value should usually be greater than 2048 to ensure
 * sufficient security.
 *
 * @param pk A pointer to an initialised egcs_public_key
 * @param vk A pointer to an initialised egcs_private_key
 * @param hr A pointer to an initialised hcs_random type
 * @param bits The number of bits for the modulus of the key
 */
void egcs_generate_key_pair(egcs_public_key *pk, egcs_private_key *vk,
                            hcs_random *hr, const unsigned long bits);

/**
 * Initialise a egcs_cipher and return a pointer to the newly created
 * structure.
 *
 * @return A pointer to an initialised egcs_cipher, NULL on allocation failure
 */
egcs_cipher* egcs_init_cipher(void);

/**
 * Copy value of op to rop.
 *
 * @param rop A pointer to an initialised egcs_cipher
 * @param op A pointer to an initialised egcs_cipher
 */
void egcs_set(egcs_cipher *rop, egcs_cipher *op);

/**
 * Encrypt a value @p plain1, and set @p rop to the encrypted result.
 *
 * @param pk A pointer to an initialised egcs_public_key
 * @param hr A pointer to an initialised hcs_random type
 * @param rop egcs_cipher where the result is to be stored
 * @param plain1 mpz_t to be encrypted
 */
void egcs_encrypt(egcs_public_key *pk, hcs_random *hr, egcs_cipher *rop,
                  mpz_t plain1);

/**
 * Multiply an encrypted value @p ct1 with an encrypted value @p ct2, storing
 * the result in @p rop.
 *
 * @param pk A pointer to an initialised egcs_public_key
 * @param rop egcs_cipher where the result is to be stored
 * @param ct1 egcs_cipher to be multiplied together
 * @param ct2 egcs_cipher to be multiplied together
 */
void egcs_ee_mul(egcs_public_key *pk, egcs_cipher *rop, egcs_cipher *ct1,
        egcs_cipher *ct2);

/**
 * Decrypt a value @p cipher1, and set @p rop to the decrypted result.
 *
 * @param vk A pointer to an initialised egcs_private_key
 * @param rop mpz_t where the decrypted result is stored
 * @param cipher1 egcs_cipher to be decrypted
 */
void egcs_decrypt(egcs_private_key *vk, mpz_t rop, egcs_cipher *cipher1);

/**
 * Zero all data related to the given egcs_cipher.
 *
 * @param ct A pointer to an initialised egcs_cipher
 */
void egcs_clear_cipher(egcs_cipher *ct);

/**
 * Free all associated memory with a given egcs_cipher.
 *
 * @param ct A pointer to an initialised egcs_cipher
 */
void egcs_free_cipher(egcs_cipher *ct);

/**
 * Zero all memory associated with a egcs_public_key. This does not free the
 * memory, so the key is safe to use immediately after calling this.
 *
 * @param pk A pointer to an initialised egcs_public_key
 */
void egcs_clear_public_key(egcs_public_key *pk);

/**
 * Zero all memory associated with a egcs_private_key. This does not free the
 * memory, so the key is safe to use immediately after calling this.
 *
 * @param vk A pointer to an initialised egcs_private_key
 */
void egcs_clear_private_key(egcs_private_key *vk);

/**
 * Frees a egcs_public_key and all associated memory.
 *
 * @param pk A pointer to an initialised egcs_public_key
 */
void egcs_free_public_key(egcs_public_key *pk);

/**
 * Frees a egcs_private_key and all associated memory.
 *
 * @param vk A pointer to an initialised egcs_private_key
 */
void egcs_free_private_key(egcs_private_key *vk);

#ifdef __cplusplus
}
#endif

#endif
