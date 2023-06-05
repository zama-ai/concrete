#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

#include <gmpxx.h>
#include "../include/libhcs++/egcs.hpp"

static hcs::random *hr;
static hcs::egcs::public_key *pk;
static hcs::egcs::private_key *vk;

TEST_CASE( "Encryption/Decryption") {

    hcs::egcs::cipher u, v;
    mpz_class a, b, c;

    /* Key pair must match */
    //REQUIRE( hcs::egcs::verify_key_pair(*pk, *vk) );

#define TEST_SIMPLE(x)\
    a = (x); b = (a);\
    u = pk->encrypt(a);\
    a = vk->decrypt(u);\
    REQUIRE( a == b )

    TEST_SIMPLE(5);
    TEST_SIMPLE(0);
    TEST_SIMPLE(1241012408124);
    TEST_SIMPLE("22222222222222222222222222222222");

#define TEST_EE_MUL(x, y)\
    a = (x); b = (y); c = (a) * (b);\
    u = pk->encrypt(a);\
    v = pk->encrypt(b);\
    u = pk->ee_mul(u, v);\
    b = vk->decrypt(u);\
    REQUIRE( b == c )

    TEST_EE_MUL(0, 0);
    TEST_EE_MUL(0, 5);
    TEST_EE_MUL(4124, 1208725);
    TEST_EE_MUL("123456678924124087124", "31209235923652352352126437357");

#undef TEST_SIMPLE
#undef TEST_EE_MUL
}

int main(int argc, char *argv[])
{
    hr = new hcs::random();
    pk = new hcs::egcs::public_key(*hr);
    vk = new hcs::egcs::private_key(*hr);
    hcs::egcs::generate_key_pair(*pk, *vk, 512);

    int result = Catch::Session().run(argc, argv);

    delete hr;
    delete pk;
    delete vk;
    return result;
}
