/**
 * @file djcs_t.h
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

#ifndef HCS_DJCS_T_HPP
#define HCS_DJCS_T_HPP

#include <string>
#include <gmpxx.h>
#include "../libhcs/djcs_t.h"
#include "random.hpp"

namespace hcs {
namespace djcs_t {

class public_key {

private:
    djcs_t_public_key *pk;
    hcs::random *hr;

public:
    public_key(hcs::random &hr_) {
        pk = djcs_t_init_public_key();
        hr = &hr_;
        hr->inc_refcount();
    }

    ~public_key() {
        djcs_free_public_key(pk);
        hr->dec_refcount();
    }

    djcs_public_key* as_ptr() {
        return pk;
    }

    hcs_random* get_rand() {
        return hr->as_ptr();
    }

    /* Encryption functions acting on a key */
    void encrypt(mpz_class &rop, mpz_class &op) {
        djcs_t_encrypt(pk, hr->as_ptr(), rop.get_mpz_t(), op.get_mpz_t());
    }

    void reencrypt(mpz_class &rop, mpz_class &op) {
        djcs_t_reencrypt(pk, hr->as_ptr(), rop.get_mpz_t(), op.get_mpz_t());
    }

    void ep_add(mpz_class &rop, mpz_class &c1, mpz_class &c2) {
        djcs_t_ep_add(pk, rop.get_mpz_t(), c1.get_mpz_t(), c2.get_mpz_t());
    }

    void ee_add(mpz_class &rop, mpz_class &c1, mpz_class &c2) {
        djcs_t_ee_add(pk, rop.get_mpz_t(), c1.get_mpz_t(), c2.get_mpz_t());
    }

    void ep_mul(mpz_class &rop, mpz_class &c1, mpz_class &p1) {
        djcs_t_ep_mul(pk, rop.get_mpz_t(), c1.get_mpz_t(), p1.get_mpz_t());
    }

    void clear() {
        djcs_t_clear_public_key(pk);
    }

    std::string export_json() {
        return std::string(djcs_t_export_public_key(pk));
    }

    int import_json(std::string &json) {
        return djcs_t_import_public_key(pk, json.c_str());
    }
};

class private_key {

private:
    djcs_t_private_key *vk;
    hcs::random *hr;

public:
    private_key(hcs::random &hr_) {
        vk = djcs_init_private_key();
        hr = &hr_;
        hr->inc_refcount();
    }

    ~private_key() {
        djcs_t_free_private_key(vk);
        hr->dec_refcount();
    }

    djcs_t_private_key* as_ptr() {
        return vk;
    }

    hcs_random* get_rand() {
        return hr->as_ptr();
    }

    void decrypt(mpz_class &rop, mpz_class &c1) {
        djcs_t_decrypt(vk, rop.get_mpz_t(), c1.get_mpz_t());
    }

    // May want to make a new object which holds shares, given there are a
    // number of specific operations that are useful to do on them, plus
    // they require a set size and we can enforce that through some function
    void share_combine(mpz_class &rop, vector<mpz_t> &shares) {
        djcs_t_share_combine(vk, rop.get_mpz_t(), shares.data());
    }

    void clear() {
        djcs_t_clear_private_key(vk);
    }

    std::string export_json() {
        return std::string(djcs_t_export_private_key(vk));
    }

    int import_json(std::string &json) {
        return djcs_t_import_private_key(vk, json.c_str());
    }
};

inline void generate_key_pair(public_key &pk, private_key &vk,
        const unsigned long bits, const unsigned long l, const unsigned long w)
{
    djcs_t_generate_key_pair(pk.as_ptr(), vk.as_ptr(), vk.get_rand(), bits,
            w, l);
}

inline int verify_key_pair(public_key &pk, private_key &vk) {
    return djcs_t_verify_key_pair(pk.as_ptr(), vk.as_ptr());
}

class polynomial {

private:
    djcs_t_polynomial *px;
    hcs::random *hr;

public:
    polynomial(hcs::random &hr_) {
        pk = djcs_t_init_polynomial();
        hr = &hr_;
        hr->inc_refcount();
    }

    ~public_key() {
        djcs_t_free_polynomial(pk);
        hr->dec_refcount();
    }

    djcs_t_poly* as_ptr() {
        return pk;
    }

    hcs_random* get_rand() {
        return hr->as_ptr();
    }

    void compute(djcs_t::private_key &vk, mpz_class &rop, const unsigned long x) {
        djcs_t_compute_polynomial(vk.as_ptr(), p, rop.get_mpz_t(), x);
    }
};

class auth_server {

private:
    djcs_t_auth_server *au;

public:
    auth_server(mpz_class &op, unsigned long id) {
        au = djcs_t_init_auth_server();
        djcs_t_set_auth_server(au, op.get_mpz_t(), id);
    }

    ~auth_server() {
        djcs_t_free_auth_server(au);
    }

    djcs_t_auth_server *as_ptr() {
        return au;
    }

    share_decrypt(public_key &pk, mpz_class &rop, mpz_class &cipher1) {
        djcs_t_share_decrypt(pk.as_ptr(), au, rop.get_mpz_t(),
                cipher1.get_mpz_t());
    }

    std::string export_json() {
        return std::string(djcs_t_export_auth_server(au));
    }

    void import_json(std::string &json) {
        djcs_t_import_auth_server(au, json.c_str());
    }
};

#ifdef __cplusplus
}
#endif

#endif
