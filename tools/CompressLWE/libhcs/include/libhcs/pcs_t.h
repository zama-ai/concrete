/**
 * @file pcs_t.h
 *
 * The threshold Paillier scheme offers the same properties as the Paillier
 * scheme, with the extra security that decryption is split performed between a
 * number of parties instead of just a single trusted party. It is much more
 * complex to set up a system which provides this, so determine if you actually
 * require this before using.
 *
 * All mpz_t values can be aliases unless otherwise stated.
 *
 * \warning All indexing for the servers and polynomial functions should be
 * zero-indexed, as is usual when working with c arrays. The functions
 * themselves correct for this internally, and 1-indexing servers may result
 * in incorrect results.
 */

#ifndef HCS_PCS_T_H
#define HCS_PCS_T_H

#include <gmp.h>
#include "hcs_random.h"
#include "hcs_shares.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Stores data pertaining to the usage of proof computation and verification.
 * This may be able to be generalised amongst all the threshold damgard-jurik
 * variants, and even the single value versions.
 */
typedef struct {
    mpz_t e[2];         /**< Internal proof verification fields */
    mpz_t a[2];         /**< Internal proof verification fields */
    mpz_t z[2];         /**< Internal proof verification fields */
    unsigned long m1;   /**< Value of first power for 1of2 protocol */
    unsigned long m2;   /**< Value of second power for 1of2 protocol */
    mpz_t generator;    /**< Value of generator (base) in 1of2 protocol */
} pcs_t_proof;

/**
 * Details of the polynomial used to compute values for decryption servers.
 */
typedef struct {
    unsigned long n;    /**< The number of terms in the polynomial */
    mpz_t *coeff;       /**< Coefficients of the polynomial */
} pcs_t_polynomial;

/**
 * Details that a decryption server is required to keep track of.
 */
typedef struct {
    unsigned long i;    /**< The server index in the particular instance */
    mpz_t si;           /**< The polynomial evaluation at @p i */
} pcs_t_auth_server;

/**
 * Public key for use in the Threshold Paillier system. This key is the main
 * key used, even during decryption we use this key AS WELL as using the
 * authorization servers.
 */
typedef struct {
    unsigned long w; /**< The number of servers req to successfully decrypt */
    unsigned long l; /**< The number of decryption servers */
    mpz_t n;         /**< Modulus of the key. n = p * q */
    mpz_t g;         /**< Precomputation: n + 1 */
    mpz_t n2;        /**< Precomputation: n^2 */
    mpz_t delta;     /**< Precomputation: l! */
} pcs_t_public_key;

/**
 * Private key for use in the Threshold Paillier system. This key is
 * effectively split up amongst parties into a number of pcs_t_auth_server
 * types. Thus, once we are done splitting this key up (computing @p l
 * pcs_t_auth_server types with pcs_t_compute_polynomial) we can safely
 * destroy this key as it will not be required again.
 */
typedef struct {
    unsigned long w; /**< The number of servers req to decrypt */
    unsigned long l; /**< The number of decryption servers in total */
    mpz_t *vi;       /**< Verification values for the decryption servers */
    mpz_t v;         /**< Cyclic generator of squares in Z*n^2 */
    mpz_t d;         /**< d = 0 mod m and d = 1 mod n^2 */
    mpz_t n;         /**< Modulus of the key: p * q */
    mpz_t n2;        /**< Precomputation: n^2 */
    mpz_t nm;        /**< Precomputation: n * m */
} pcs_t_private_key;

/**
 * Initialise a pcs_t_public_key and return a pointer to the newly created
 * structure.
 *
 * @return A pointer to an initialised pcs_t_public_key, NULL on allocation
 *         failure
 */
pcs_t_public_key* pcs_t_init_public_key(void);

/**
 * Initialise a pcs_t_private_key and return a pointer to the newly created
 * structure.
 *
 * @return A pointer to an initialised pcs_t_private_key, NULL on allocation
 *         failure
 */
pcs_t_private_key* pcs_t_init_private_key(void);

/**
 * Initialise a key pair with modulus size @p bits. It is required that @p pk
 * and @p vk are initialised before calling this function. @p pk and @p vk are
 * expected to not be NULL. @p w is the number of servers total to be used with
 * the resulting keys. @p l is the number of servers required to succesfully
 * decrypt a given message.
 *
 * In practice the @p bits value should usually be greater than 2048 to ensure
 * sufficient security.
 *
 * \warning This function attempts to allocate memory, so calling this twice in
 * succession with the same keys will cause your program to lose a pointer to
 * this allocated memory, resulting in a memory leak. If you wish to call this
 * function in this manner, ensure pcs_t_clear_public_key and/or
 * pcs_t_clear_private_key are called prior.
 *
 * \pre 0 < @p l <= @p w
 *
 * @code
 * pcs_t_public_key *pk = pcs_t_init_public_key();
 * pcs_t_private_key *vk = pcs_t_init_private_key();
 * hcs_random = hcs_random_init();
 * pcs_t_generate_key(pk, vk, hr, 2048, 5, 7);
 * @endcode
 *
 * @param pk A pointer to an initialised pcs_t_public_key
 * @param vk A pointer to an initialised pcs_t_private_key
 * @param hr A pointer to an initialised hcs_random type
 * @param bits The number of bits for the modulus of the key
 * @param l The number of servers required to succesfully decrypt
 * @param w The number of servers in total
 * @return non-zero on success, zero on allocation failure
 */
int pcs_t_generate_key_pair(pcs_t_public_key *pk, pcs_t_private_key *vk,
        hcs_random *hr, const unsigned long bits, const unsigned long l,
        const unsigned long w);

/**
 * Encrypt a value @p plain1, and set @p rop to the encryted result. This
 * function uses the random value @p r, passed as a parameter by the caller
 * instead of generating a value with an hcs_random object.
 *
 * @p r should be in the field Z_{n^2}*.
 *
 * @param pk A pointer to an initialised pcs_t_public_key
 * @param r mpz_t where the random value to use is stored
 * @param rop mpz_t where the encrypted result is stored
 * @param plain1 mpz_t to be encrypted
 */
void pcs_t_encrypt_r(pcs_t_public_key *pk, mpz_t rop, mpz_t r, mpz_t plain1);

/**
 * Encrypt a value @p plain1, and set @p rop to the encrypted result. This only
 * differs from pcs_t_encrypt in that the random value generated for use in
 * this encryption is stored in the mpz_t variable, @p r.
 *
 * @param pk A pointer to an initialised pcs_t_public_key
 * @param hr A pointer to an initialised hcs_random type
 * @param r mpz_t where the random value generated is stored
 * @param rop mpz_t where the encrypted result is stored
 * @param plain1 mpz_t to be encrypted
 */
void pcs_t_r_encrypt(pcs_t_public_key *pk, hcs_random *hr,
        mpz_t r, mpz_t rop, mpz_t plain1);

/**
 * Encrypt a value @p plain1, and set @p rop to the encrypted result.
 *
 * @param pk A pointer to an initialised pcs_t_public_key
 * @param hr A pointer to an initialised hcs_random type
 * @param rop mpz_t where the encrypted result is stored
 * @param plain1 mpz_t to be encrypted
 */
void pcs_t_encrypt(pcs_t_public_key *pk, hcs_random *hr, mpz_t rop, mpz_t plain1);

/**
 * Reencrypt an encrypted value @p cipher1. Upon decryption, this newly
 * encrypted value, @p rop, will retain the same value as @p op.
 *
 * @param pk A pointer to an initialised pcs_t_public_key
 * @param hr A pointer to an initialised hcs_random type
 * @param rop mpz_t where the newly encrypted value is stored
 * @param op mpz_t to be reencrypted
 */
void pcs_t_reencrypt(pcs_t_public_key *pk, hcs_random *hr, mpz_t rop, mpz_t op);

/**
 * Add a plaintext value @p plain1 to an encrypted value @p cipher1, storing
 * the result in @p rop.
 *
 * @param pk A pointer to an initialised pcs_t_public_key
 * @param rop mpz_t where the newly encrypted value is stored
 * @param cipher1 mpz_t to be added together
 * @param plain1 mpz_t to be added together
 */
void pcs_t_ep_add(pcs_t_public_key *pk, mpz_t rop, mpz_t cipher1, mpz_t plain1);

/**
 * Add an encrypted value @p cipher2 to an encrypted value @p cipher1, storing
 * the result in @p rop.
 *
 * @param pk A pointer to an initialised pcs_t_public_key.
 * @param rop mpz_t where the newly encrypted value is stored
 * @param cipher1 mpz_t to be added together
 * @param cipher2 mpz_t to be added together
 */
void pcs_t_ee_add(pcs_t_public_key *pk, mpz_t rop, mpz_t cipher1, mpz_t cipher2);

/**
 * Multiply a plaintext value @p plain1 with an encrypted value @p cipher1,
 * storing the result in @p rop.
 *
 * @param pk A pointer to an initialised pcs_t_public_key
 * @param rop Where the new encrypted result is stored
 * @param cipher1 The encrypted value which is to be multipled to
 * @param plain1 The plaintext value which is to be multipled
 */
void pcs_t_ep_mul(pcs_t_public_key *pk, mpz_t rop, mpz_t cipher1, mpz_t plain1);

/**
 * Allocate and initialise the values in a pcs_t_proof object. This is used
 * in all verification and computation involving proofs.
 *
 * @return A zero-initialised pcs_t_proof object
 */
pcs_t_proof* pcs_t_init_proof(void);

/**
 * Set a proof object's value to check for.
 *
 * @param pf A pointer to an initialised pcs_t_proof object
 * @param generator Value to set the generator of the proof
 * @param m1 First power that this proof will allow
 * @param m2 Second power that this proof will allow
 */
void pcs_t_set_proof(pcs_t_proof *pf, mpz_t generator, unsigned long m1,
        unsigned long m2);

/**
 * Compute a proof for a particular value @p ciper_m, and an @p id. @p cipher_m
 * is the encrypted value for which the proof is generated for, with @p cipher_r
 * being the random value used to generate it. @p cipher_m and @p cipher_r
 * can be generated using the pcs_t_r_encrypt function.
 *
 * The result of this function is a proof result stored in @p pf, which can be
 * queried to determine if @p cipher_m is an encryption of an n'th power.
 *
 * @param pk A pointer to an initialised pcs_t_public_key
 * @param hr A pointer to an initialised hcs_random object
 * @param pf A pointer to an initialised pcs_t_proof object
 * @param cipher_m The encrypted value
 * @param cipher_r The random r value used for this cipher text
 * @param nth_power The power that this encrypted value represents
 * @param id User id in the system. This can be discarded by using the value 0
 */
void pcs_t_compute_1of2_ns_protocol(pcs_t_public_key *pk, hcs_random *hr,
        pcs_t_proof *pf, mpz_t cipher_m, mpz_t cipher_r, unsigned long nth_power,
        unsigned long id);

/**
 * Verify a proof and return whether it is an n'th power.
 *
 * @param pk A pointer to an initialised pcs_t_public_key
 * @param pf A pointer to an initialised hcs_random object
 * @param id User id in the system.
 * @return 0 if not an n'th power, non-zero if it is an n'th power
 */
int pcs_t_verify_ns_protocol(pcs_t_public_key *pk, pcs_t_proof *pf,
        unsigned long id);

/**
 * Compute the value for an n^s protocol, limited to 1 of 2 values.
 *
 * @param pk A pointer to an initialised pcs_t_public_key
 * @param pf A pointer to an initialised pcs_t_proof
 * @param cipher Encrypted value that this proof is meant to verify
 * @param id of the owner of this cipher text
 */
int pcs_t_verify_1of2_ns_protocol(pcs_t_public_key *pk, pcs_t_proof *pf,
        mpz_t cipher, unsigned long id);

/**
 * Frees a pcs_t proof object and all values associated with it.
 *
 * @param pf An initialised pcs_t_proof object
 */
void pcs_t_free_proof(pcs_t_proof *pf);

/**
 * Allocate and initialise the values in a random polynomial. The length of
 * this polynomial is taken from values in @p vk, specifically it will be
 * of length vk->w. The polynomial functions are to be used by a single trusted
 * party, for which once the required computation is completed, the polynomial
 * can be discarded.
 *
 * @code
 * mpz_t *poly = pcs_t_init_polynomial(vk, hr);
 * for (int i = 0; i < decrypt_server_count; ++i) {
 *     pcs_t_compute_polynomial(vk, poly, result, i);
 *     network_send(result); // Send the computed value to a decryption server
 * }
 * pcs_t_free_polynomial(vk, poly);
 * @endcode
 *
 * @param vk A pointer to an initialised pcs_t_private_key
 * @param hr A pointer to an initialised hcs_random type
 * @return A polynomial coefficient list on success, else NULL
 */
pcs_t_polynomial* pcs_t_init_polynomial(pcs_t_private_key *vk, hcs_random *hr);

/**
 * Compute a polynomial P(x) for a given x value in the required finite field.
 * The coefficients should be given as a list of mpz_t values, computed via the
 * pcs_t_init_polynomial function.
 *
 * @param vk A pointer to an initialised pcs_t_private_key
 * @param px A pointer to a list of coefficients of a polynomial
 * @param rop mpz_t where the result is stored
 * @param x The value to calculate the polynomial at
 */
void pcs_t_compute_polynomial(pcs_t_private_key *vk, pcs_t_polynomial *px,
                              mpz_t rop, const unsigned long x);

/**
 * Frees a given polynomial (array of mpz_t values) and all associated data.
 * The same private key which was used to generate these values should be used
 * as an argument.
 *
 * @param px A pointer to a pcs_t_polynomial
 */
void pcs_t_free_polynomial(pcs_t_polynomial *px);

/**
 * Initialise a pcs_t_auth_server and return a pointer to the newly created
 * structure.
 *
 * @return A pointer to an initialised pcs_t_auth_server, NULL on allocation
 *         failure
 */
pcs_t_auth_server* pcs_t_init_auth_server(void);

/**
 * Set the internal values for the server @p au. @p si is the secret polynomial
 * share for the given value, @p i. These values should be shared in a secret
 * and secure way and not given out publicly. The index given to each server
 * should be unique.
 *
 * @param au A pointer to an initialised pcs_t_auth_server
 * @param si The value of a secret polynomial evaluated at @p i
 * @param i The servers given index
 */
void pcs_t_set_auth_server(pcs_t_auth_server *au, mpz_t si, unsigned long i);

/**
 * For a given ciphertext @p cipher1, compute the server @p au's share and store
 * the result in the variable @p rop. These shares can be managed, and then
 * combined when sufficient shares have been accumulated using the
 * pcs_t_share_combine function.
 *
 * @param vk A pointer to an initialised pcs_t_private_key
 * @param au A pointer to an initialised pcs_t_auth_server
 * @param rop mpz_t where the calculated share is stored
 * @param cipher1 mpz_t which stores the ciphertext to decrypt
 */
void pcs_t_share_decrypt(pcs_t_public_key *vk, pcs_t_auth_server *au,
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
 * @param vk A pointer to an initialised pcs_t_private_key
 * @param rop mpz_t where the combined decrypted result is stored
 * @param hs A pointer to an initialised hcs_shares
 */
int pcs_t_share_combine(pcs_t_public_key *vk, mpz_t rop, hcs_shares *hs);

/**
 * Frees a pcs_t_auth_server and all associated memory.
 *
 * @param au A pointer to an initialised pcs_t_auth_server
 */
void pcs_t_free_auth_server(pcs_t_auth_server *au);

/**
 * Clears all data in a pcs_t_public_key. This does not free memory in the
 * keys, only putting it into a state whereby they can be safely used to
 * generate new key values.
 *
 * @param pk A pointer to an initialised pcs_t_public_key
 */
void pcs_t_clear_public_key(pcs_t_public_key *pk);

/**
 * Clears all data in a pcs_t_private_key. This does not free memory in the
 * keys, only putting it into a state whereby they can be safely used to
 * generate new key values.
 *
 * @param vk A pointer to an initialised pcs_t_private_key
 */
void pcs_t_clear_private_key(pcs_t_private_key *vk);

/**
 * Frees a pcs_t_public_key and all associated memory. The key memory is
 * not zeroed, so one must call pcs_t_clear_public_key if it is required.
 * one does not need to call pcs_t_clear_public_key before using this function.
 *
 * @param pk A pointer to an initialised pcs_t_public_key
 */
void pcs_t_free_public_key(pcs_t_public_key *pk);

/**
 * Frees a pcs_t_private_key and all associated memory. The key memory is
 * not zeroed, so one must call pcs_t_clear_private_key if it is required.
 * one does not need to call pcs_t_clear_private_key before using this function.
 *
 * @param vk A pointer to an initialised pcs_t_private_key
 */
void pcs_t_free_private_key(pcs_t_private_key *vk);

/**
 * Sanity check to quickly determine if pk and vk refer to the same values and
 * are compatible.
 *
 * @param pk A pointer to an initialised pcs_t_public_key
 * @param vk A pointer to an initialised pcs_t_private_key
 * @return non-zero if keys match, zero if they do not
 */
int pcs_t_verify_key_pair(pcs_t_public_key *pk, pcs_t_private_key *vk);

/**
 * Import a public key from a json string. The format of the json string must
 * match those of the export functions, and is described more in-depth at
 * @(refer to description of serialization).
 *
 * @param pk A pointer to an initialised pcs_t_public_key
 * @param json A null-terminated json string containing public key data
 * @return 0 on success, non-zero on parse failure
 */
int pcs_t_import_public_key(pcs_t_public_key *pk, const char *json);

/**
 * Export a public key from as a json string. This function allocates memory
 * and it is up to the caller to free it accordingly.
 *
 * @param pk A pointer to an initialised pcs_t_public_key
 * @return A null-terminated json string
 */
char *pcs_t_export_public_key(pcs_t_public_key *pk);

/**
 * Export an auth server as a json string. This function allocates memory and
 * it is up to the caller to free it accordingly.
 *
 * @param au A pointer to an initialised pcs_t_auth_server
 * @return A null-terminated json string
 */
char *pcs_t_export_auth_server(pcs_t_auth_server *au);

/**
 * Import an auth server from a json string. The format of the json string
 * must match that of the corresponding import function.
 *
 * @param au A pointer to an initialised pcs_t_auth_server
 * @param json A null-terminated json string containing auth server data
 * @return 0 on success, non-zero on parse failure
 */
int pcs_t_import_auth_server(pcs_t_auth_server *au, const char *json);

/**
 * Export a proof as a json string. This function allocates memory and it is
 * up to the caller to free it accordingly.
 *
 * @param pf A pointer to an initialised pcs_t_proof
 * @return A null-terminated json string
 */
char *pcs_t_export_proof(pcs_t_proof *pf);

/**
 * Import pcs_t_proof data from a json string. The format of the json string
 * must match that of the corresponding import function.
 *
 * @param pf A pointer to an initialised pcs_t_proof
 * @param json A null-terminated json string containing pcs_t_proof data
 * @return 0 on success, non-zero on parse failure
 */
int pcs_t_import_proof(pcs_t_proof *pf, const char *json);

/**
 * Export an array of verification values corresponding to each server as
 * a json array. This is seperate from the private key export function as it
 * may not always be wanted.
 *
 * @warning Unimplemented
 *
 * @todo This function will eventually be altered so that the verification
 * values are stored in a pcs_t_public_key.
 *
 * @param vk A pointer to an initialised pcs_t_private_key
 * @return json A null-terminated json string containing all server
 *              verification data. NULL is returned on case of parse error.
 */
char *pcs_t_export_verify_values(pcs_t_private_key *vk);

/**
 * Import an array of verification values stored in a json string. The format
 * should matched that produced by pcs_t_export_verify_values.
 *
 * @warning Unimplemented
 *
 * @todo As with the export function
 *
 * @param vk A pointer to an initialised pcs_t_private_key
 * @param json A null-terminated json string containing verification values
 * @return non-zero on success and 0 on failure
 */
int pcs_t_import_verify_values(pcs_t_private_key *vk, const char *json);

#ifdef __cplusplus
}
#endif

#endif
