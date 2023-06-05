/**
 * @file djcs.hpp
 *
 * A C++ wrapper class around djcs.c
 */

#ifndef HCS_DJCS_HPP
#define HCS_DJCS_HPP

#include <string.h>
#include <gmpxx.h>
#include "../libhcs/djcs.h"
#include "random.hpp"

/*
 * We do not manage the memory associated with an hcs::random class here, and it
 * is up to the caller to ensure that the hcs::random associated has the same
 * lifetime as the public/private key.
 */

namespace hcs {
namespace djcs {

class public_key {

private:
    djcs_public_key *pk;
    hcs::random *hr;

public:
    public_key(hcs::random &hr_) {
        pk = djcs_init_public_key();
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
        djcs_encrypt(pk, hr->as_ptr(), rop.get_mpz_t(), op.get_mpz_t());
    }

    void reencrypt(mpz_class &rop, mpz_class &op) {
        djcs_reencrypt(pk, hr->as_ptr(), rop.get_mpz_t(), op.get_mpz_t());
    }

    void ep_add(mpz_class &rop, mpz_class &c1, mpz_class &c2) {
        djcs_ep_add(pk, rop.get_mpz_t(), c1.get_mpz_t(), c2.get_mpz_t());
    }

    void ee_add(mpz_class &rop, mpz_class &c1, mpz_class &c2) {
        djcs_ee_add(pk, rop.get_mpz_t(), c1.get_mpz_t(), c2.get_mpz_t());
    }

    void ep_mul(mpz_class &rop, mpz_class &c1, mpz_class &p1) {
        djcs_ep_mul(pk, rop.get_mpz_t(), c1.get_mpz_t(), p1.get_mpz_t());
    }

    void clear() {
        djcs_clear_public_key(pk);
    }

    std::string export_json() {
        return std::string(djcs_export_public_key(pk));
    }

    int import_json(std::string &json) {
        return djcs_import_public_key(pk, json.c_str());
    }
};

class private_key {

private:
    djcs_private_key *vk;
    hcs::random *hr;

public:
    private_key(hcs::random &hr_) {
        vk = djcs_init_private_key();
        hr = &hr_;
        hr->inc_refcount();
    }

    ~private_key() {
        djcs_free_private_key(vk);
        hr->dec_refcount();
    }

    djcs_private_key* as_ptr() {
        return vk;
    }

    hcs_random* get_rand() {
        return hr->as_ptr();
    }

    void decrypt(mpz_class &rop, mpz_class &c1) {
        djcs_decrypt(vk, rop.get_mpz_t(), c1.get_mpz_t());
    }

    void clear() {
        djcs_clear_private_key(vk);
    }

    std::string export_json() {
        return std::string(djcs_export_private_key(vk));
    }

    int import_json(std::string &json) {
        return djcs_import_private_key(vk, json.c_str());
    }
};

inline void generate_key_pair(public_key &pk, private_key &vk,const unsigned long s,
        const unsigned long bits)
{
    djcs_generate_key_pair(pk.as_ptr(), vk.as_ptr(), vk.get_rand(), s, bits);
}

inline int verify_key_pair(public_key &pk, private_key &vk) {
    return djcs_verify_key_pair(pk.as_ptr(), vk.as_ptr());
}

} // djcs namespace
} // hcs namespace
#endif
