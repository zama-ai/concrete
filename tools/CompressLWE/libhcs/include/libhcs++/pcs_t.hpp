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

#ifndef HCS_PCS_T_HPP
#define HCS_PCS_T_HPP

#include <string>
#include <vector>
#include <gmpxx.h>
#include "../libhcs/pcs_t.h"
#include "random.hpp"

namespace hcs {
namespace pcs_t {

class public_key {

private:
    pcs_t_public_key *pk;
    hcs::random *hr;

public:
    public_key(hcs::random &hr_) {
        pk = pcs_t_init_public_key();
        hr = &hr_;
        hr->inc_refcount();
    }

    ~public_key() {
        pcs_t_free_public_key(pk);
        hr->dec_refcount();
    }

    pcs_t_public_key* as_ptr() {
        return pk;
    }

    hcs::random* get_rand() {
        return hr;
    }

    /* Encryption functions acting on a key */
    mpz_class encrypt(mpz_class &op) {
        mpz_class rop;
        pcs_t_encrypt(pk, hr->as_ptr(), rop.get_mpz_t(), op.get_mpz_t());
        return rop;
    }

    mpz_class reencrypt(mpz_class &op) {
        mpz_class rop;
        pcs_t_reencrypt(pk, hr->as_ptr(), rop.get_mpz_t(), op.get_mpz_t());
        return rop;
    }

    mpz_class ep_add(mpz_class &c1, mpz_class &c2) {
        mpz_class rop;
        pcs_t_ep_add(pk, rop.get_mpz_t(), c1.get_mpz_t(), c2.get_mpz_t());
        return rop;
    }

    mpz_class ee_add(mpz_class &c1, mpz_class &c2) {
        mpz_class rop;
        pcs_t_ee_add(pk, rop.get_mpz_t(), c1.get_mpz_t(), c2.get_mpz_t());
        return rop;
    }

    mpz_class ep_mul(mpz_class &c1, mpz_class &p1) {
        mpz_class rop;
        pcs_t_ep_mul(pk, rop.get_mpz_t(), c1.get_mpz_t(), p1.get_mpz_t());
        return rop;
    }

    // May want to make a new object which holds shares, given there are a
    // number of specific operations that are useful to do on them, plus
    // they require a set size and we can enforce that through some function
    mpz_class share_combine(std::vector<mpz_t> &shares) {
        mpz_class rop;
        pcs_t_share_combine(pk, rop.get_mpz_t(), shares.data());
        return rop;
    }

    void clear() {
        pcs_t_clear_public_key(pk);
    }

    std::string export_json() {
        return std::string(pcs_t_export_public_key(pk));
    }

    int import_json(std::string &json) {
        return pcs_t_import_public_key(pk, json.c_str());
    }
};

class private_key {

private:
    pcs_t_private_key *vk;
    hcs::random *hr;

public:
    private_key(hcs::random &hr_) {
        vk = pcs_t_init_private_key();
        hr = &hr_;
        hr->inc_refcount();
    }

    ~private_key() {
        pcs_t_free_private_key(vk);
        hr->dec_refcount();
    }

    pcs_t_private_key* as_ptr() {
        return vk;
    }

    hcs::random* get_rand() {
        return hr;
    }

    void decrypt(mpz_class &rop, mpz_class &c1) {
        //pcs_t_decrypt(vk, rop.get_mpz_t(), c1.get_mpz_t());
    }

    void clear() {
        pcs_t_clear_private_key(vk);
    }

    std::string export_json() {
        //return std::string(pcs_t_export_private_key(vk));
    }

    int import_json(std::string &json) {
        //return pcs_t_import_private_key(vk, json.c_str());
    }
};

class polynomial {

private:
    pcs_t_polynomial *px;
    hcs::random *hr;

public:
    polynomial(hcs::pcs_t::private_key &pk) {
        hr = pk.get_rand();
        px = pcs_t_init_polynomial(pk.as_ptr(), hr->as_ptr());
        hr->inc_refcount();
    }

    ~polynomial() {
        pcs_t_free_polynomial(px);
        hr->dec_refcount();
    }

    pcs_t_polynomial* as_ptr() {
        return px;
    }

    hcs::random* get_rand() {
        return hr;
    }

    mpz_class compute(pcs_t::private_key &vk, const unsigned long x) {
        mpz_class rop;
        pcs_t_compute_polynomial(vk.as_ptr(), px, rop.get_mpz_t(), x);
        return rop;
    }
};

class auth_server {

private:
    pcs_t_auth_server *au;

public:
    auth_server(mpz_class &op, unsigned long id) {
        au = pcs_t_init_auth_server();
        pcs_t_set_auth_server(au, op.get_mpz_t(), id);
    }

    ~auth_server() {
        pcs_t_free_auth_server(au);
    }

    pcs_t_auth_server *as_ptr() {
        return au;
    }

    mpz_class share_decrypt(public_key &pk, mpz_class &cipher1) {
        mpz_class rop;
        pcs_t_share_decrypt(pk.as_ptr(), au, rop.get_mpz_t(),
                cipher1.get_mpz_t());
        return rop;
    }

    std::string export_json() {
        return std::string(pcs_t_export_auth_server(au));
    }

    void import_json(std::string &json) {
        pcs_t_import_auth_server(au, json.c_str());
    }
};

inline void generate_key_pair(public_key &pk, private_key &vk,
        const unsigned long bits, const unsigned long l, const unsigned long w)
{
    pcs_t_generate_key_pair(pk.as_ptr(), vk.as_ptr(), vk.get_rand()->as_ptr(),
                            bits, w, l);
}

inline int verify_key_pair(public_key &pk, private_key &vk) {
    return pcs_t_verify_key_pair(pk.as_ptr(), vk.as_ptr());
}

}
}

#endif
