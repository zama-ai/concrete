#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

#include <gmpxx.h>
#include "../include/libhcs++/pcs.hpp"

static hcs::random *hr;
static hcs::pcs::public_key *pk;
static hcs::pcs::private_key *vk;

TEST_CASE( "Encryption/Decryption") {
    mpz_class a, b, c, d;

    /* Key pair must match */
    REQUIRE( hcs::pcs::verify_key_pair(*pk, *vk) );

#define TEST_SIMPLE(x)\
    a = (x); d = (a);\
    a = pk->encrypt(a);\
    a = vk->decrypt(a);\
    REQUIRE( a == d )

    TEST_SIMPLE(5);
    TEST_SIMPLE(0);
    TEST_SIMPLE(124100124);
    TEST_SIMPLE("22222222222222222222222222222222");

#define TEST_EP_ADD(x, y)\
    a = (x); b = (y); d = (a) + (b);\
    a = pk->encrypt(a);\
    a = pk->ep_add(a, b);\
    c = vk->decrypt(a);\
    REQUIRE( c == d )

    TEST_EP_ADD(0, 0);
    TEST_EP_ADD(0, 5);
    TEST_EP_ADD(4124, 1208725);
    TEST_EP_ADD("123456678924124087124", "31209235923652352352126437357");

#define TEST_EE_ADD(x, y)\
    a = (x); b = (y); d = (a) + (b);\
    a = pk->encrypt(a);\
    b = pk->encrypt(b);\
    a = pk->ee_add(a, b);\
    c = vk->decrypt(a);\
    REQUIRE( c == d )

    TEST_EE_ADD(0, 0);
    TEST_EE_ADD(0, 5);
    TEST_EE_ADD(4124, 1208725);
    TEST_EE_ADD("123456678924124087124", "31209235923652352352126437357");

#define TEST_EP_MUL(x, y)\
    a = (x); b = (y); d = (a) * (b);\
    a = pk->encrypt(a);\
    a = pk->ep_mul(a, b);\
    c = vk->decrypt(a);\
    REQUIRE( c == d )

    TEST_EP_MUL(0, 0);
    TEST_EP_MUL(0, 5);
    TEST_EP_MUL(4124, 1208725);
    TEST_EP_MUL("123456678924124087124", "31209235923652352352126437357");

#define TEST_REENCRYPT(x)\
    a = (x); d = (x);\
    a = pk->encrypt(a);\
    a = pk->reencrypt(a);\
    a = vk->decrypt(a);\
    REQUIRE( a == d )

    TEST_REENCRYPT(5);
    TEST_REENCRYPT(0);
    TEST_REENCRYPT(124101284);
    TEST_REENCRYPT("22222222222222222222222222222222");

#undef TEST_SIMPLE
#undef TEST_EP_ADD
#undef TEST_EE_ADD
#undef TEST_EP_MUL
#undef TEST_REENCRYPT
}

int main(int argc, char *argv[])
{
    hr = new hcs::random();
    pk = new hcs::pcs::public_key(*hr);
    vk = new hcs::pcs::private_key(*hr);
    hcs::pcs::generate_key_pair(*pk, *vk, 512);

    int result = Catch::Session().run(argc, argv);

    delete hr;
    delete pk;
    delete vk;
    return result;
}
