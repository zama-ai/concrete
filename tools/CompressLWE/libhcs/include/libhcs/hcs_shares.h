/**
 * @file hcs_shares.h
 *
 * This is a structure which is designed to hold a number of shares. It can be
 * thought of as an array of mpz_t values, each with a corresponding bit which
 * signals if the particular value of the share should be counted when
 * combining shares. This is used by all threshold encryption schemes.
 *
 * The rationale for this is that one may want to test a number of different
 * combinations of shares, and setting a flag simplifies this process.
 */

#ifndef HCS_SHARES
#define HCS_SHARES

#include <gmp.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Stores a number of shares, with flags indicating if they are currently to
 * be counted.
 */
typedef struct {
    mpz_t *shares;      /**< An array of share values */
    int *flag;          /**< Flags corresponding to each value in @p shares */
    void **server_id;   /**< Optional id of each shares server */
    unsigned long size; /**< Number of shares in this array */
} hcs_shares;

/**
 * Initialise a hcs_shares and return a pointer to the newly created structure.
 *
 * @param size The number of shares this hcs_shares should store
 * @return A pointer to an initialised hcs_shares, NULL on allocation failure
 */
hcs_shares* hcs_init_shares(unsigned long size);

/**
 * Set a share value for the server given by id @p index. @p index should
 * be less than @p hs->size. It is up to the caller to enforce this. The
 * owner of this @p hs should have a mapping of server id's to indices, as it
 * is required by a number of other functions involved in threshold system.
 *
 * @param hs A pointer to an initialised hcs_shares
 * @param value Share stored in an mpz_t variable
 * @param index Index of the server to store this share in
 */
void hcs_set_share(hcs_shares *hs, mpz_t value, unsigned long index);

/**
 * Set the flag on @p hs at @p index. This share will then be counted by
 * any share combinining functions that are subsequently called.
 *
 * @param hs A pointer to an initialised hcs_shares
 * @param index Index of the server to set
 */
void hcs_set_flag(hcs_shares *hs, unsigned long index);

/**
 * Clear the flag on @p hs at @p index. This share will NOT be counted by
 * any share combining functions that are subsequently called.
 *
 * @param hs A pointer to an initialised hcs_shares
 * @param index Index of the server to clear
 */
void hcs_clear_flag(hcs_shares *hs, unsigned long index);

/**
 * Toggle the flag on @p hs at @p index.
 *
 * @param hs A pointer to an initialised hcs_shares
 * @param index Index of the server to clear
 */
void hcs_toggle_flag(hcs_shares *hs, unsigned long index);

/**
 * Test if the server given by @p index is to be counted.
 *
 * @param hs A pointer to an initialised hcs_shares
 * @param index Index of the server to clear
 * @return non-zero if flag is set, 0 otherwise
 */
int hcs_tst_flag(hcs_shares *hs, unsigned long index);

/**
 * Frees a hcs_shares and all associated memory.
 *
 * @param hs A pointer to an initialised hcs_shares
 */
void hcs_free_shares(hcs_shares *hs);

#ifdef __cplusplus
}
#endif

#endif

