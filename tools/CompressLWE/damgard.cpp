#include <iostream>
#include <vector>
#include <gmp.h>    // gmp is included implicitly
#include <libhcs++.hpp> // master header includes everything
#include "defines.h"
using namespace std;

int main(){

    uint64_t n=630;
    uint64_t log_q=64;

    // ct = (a_0,b)
    vector<uint64_t> ct(n+1, 1);

    vector<uint64_t> sk(n, 1);
//////////////////////////////
    hcs::random hr;;

    // initialize data structures
    hcs::djcs::public_key pk(hr);
    hcs::djcs::private_key vk(hr);
    size_t s = 3;
    // Generate a key pair with modulus of size 2048 bits
    hcs::djcs::generate_key_pair(pk, vk, s, 2048);


    mpz_class a (18446744073709551616_mpz);
    mpz_class b (10203040506070809010020030040050060070080090_mpz);
    gmp_printf("a = %Zd\nb = %Zd\n", a.get_mpz_t(), b.get_mpz_t());
    mpz_class scale (14741117345057607444_mpz);
    pk.encrypt(a, a);
    pk.encrypt(b, b);
    gmp_printf("a = %Zd\nb = %Zd\n", a.get_mpz_t(), b.get_mpz_t()); // can use all gmp functions still
    mpz_class c;
    pk.ee_add(c, a, b);
    vk.decrypt(c, c);
    gmp_printf("%Zd\n", c.get_mpz_t());
    mpz_vec as(50);
    mpz_vec bs(50);
    mpz_vec cs(50);
    for (int i = 0; i < 50; ++i) {
        as[i]=i+1;
        bs[i] = i;
        pk.encrypt(as[i],as[i]);
        pk.ep_mul(cs[i], as[i], bs[i]);
        vk.decrypt(cs[i], cs[i]);
        gmp_printf("%Zd\n", cs[i].get_mpz_t());
    }
    pk.ep_mul(c, a, scale);
    vk.decrypt(c, c);
    gmp_printf("%Zd\n", c.get_mpz_t());
    return 0;
}