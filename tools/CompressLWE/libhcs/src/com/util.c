/*
 * @file util.c
 */

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdint.h>
#include <gmp.h>
#include "ripemd160.h"
#include "util.h"

#ifdef _WIN32
#include "Wincrypt.h"                  // For CryptGenRandom
#pragma comment(lib, "advapi32.lib")
#endif

/* If we are compiling with C11 support, we will have memset_s, otherwise
 * use this simple version instead. */
#if 0
#if __STDC_VERSION__ < 201112L
static void memset_s(void *v, int c, size_t n)
{
    volatile unsigned char *p = v;
    while (n--) *p++ = c;
}
#else
#include <string.h>
#endif
#endif

/* Zero a single mpz_t variable. mpz_clear does not seem to be required
 * to zero memory before freeing it, so we must do it ourselves. A
 * function mpn_zero exists which performs this. */
inline void mpz_zero(mpz_t op)
{
    mpn_zero(op->_mp_d, op->_mp_alloc);
}

/* Call mpz_zero with a va_list. This follows mpz_clears and mpz_inits
 * in that it is a NULL-terminated va_list */
void mpz_zeros(mpz_t op, ...)
{
    va_list va;
    va_start(va, op);

    /* Use the internal mpz names pre-typedef so we can assign pointers, as
     * mpz_t is defined as __mpz_struct[1] which is non-assignable */
    __mpz_struct *ptr = op;
    do {
        mpz_zero(ptr);
    } while ((ptr = va_arg(va, __mpz_struct*)));
}

/* Attempts to get n bits of seed data from /dev/urandom. The number of
 * bits is always round up to the nearest 8. Asking for 78 bits of seed
 * will gather 80 bits, for example. */
int mpz_seed(mpz_t seed, int bits)
{
#if defined(__unix__) || defined(__linux__) || (defined(__APPLE__) && defined(__MACH__))
    FILE *fd = fopen("/dev/urandom", "rb");
    if (!fd)
        return HCS_EOPEN;

    const int bytes = (bits / 8) + 1;
    unsigned char random_bytes[bytes];
    if (fread(random_bytes, sizeof(random_bytes), 1, fd) != 1)
        return HCS_EREAD;

    mpz_import(seed, bytes, 1, sizeof(random_bytes[0]), 0, 0, random_bytes);
    //memset_s(random_bytes, 0, bytes); /* Ensure we zero seed buffer data */
    fclose(fd);

#elif defined(_WIN32)
    const DWORD bytes = (bits / 8) + 1;
    HCRYPTPROV hCryptProv = 0;
    BYTE pbBuffer[bytes];

    if (!CryptAcquireContextW(&hCryptProv, 0, 0, PROV_RSA_FULL,
            CRYPT_VERIFYCONTEXT, CRYPT_SILENT) {
       return HCS_EREAD;
    }

    if (!CryptGenRandom(hCryptProv, bytes, pbBuffer)) {
        CryptReleaseContext(hCryptProv, 0);
        return HCS_EREAD;
    }

    mpz_import(seed, bytes, 1, sizeof(pbBuffer[0]), 0, 0, pbBuffer);
    //memset_s(pbBuffer, 0, bytes); /* Ensure we zero seed buffer data */

    if (!CryptReleaseContext(hCryptProv, 0)) {
        return HCS_EREAD;
    }
#else
#   error "No random source known for this OS"
#endif

    return HCS_OK;
}

/* Generate a random value that is in Z_(op)^*. This simply random chooses
 * values until we get one with gcd(rop, op) of n. If one has knowledge about
 * the value of rop, then calling this function may not be neccessary. i.e.
 * if rop is prime, we can just call urandomm directly. */
void mpz_random_in_mult_group(mpz_t rop, gmp_randstate_t rstate, mpz_t op)
{
    mpz_t t1;
    mpz_init(t1);

    do {
        mpz_urandomm(rop, rstate, op);
        mpz_gcd(t1, rop, op);
    } while (mpz_cmp_ui(t1, 1) != 0);

    mpz_clear(t1);
}

// TODO: This is uneeded until generator_g_compute is implemented. This is
// also only an alternative to generator_gd_compute, and may not be implemented
// at all.
#if 0
static void generator_g_precompute(mpz_t pi, mpz_t t, mpz_t theta, mp_bitcnt_t bitcnt)
{
    /* This function generates a value, pi, which minimizes the ratio
     * phi(pi) / pi by choosing all prime factors of small powers. Also,
     * we return a value t, which is 2 * p^k, where p^k is the maximum value
     * of any prime power in the value pi. Further, we construct a sequence
     * theta, which represents a bit pattern corresponding to pi's prime
     * factors. */

    mpz_t p;
    mpz_init(p);

    mpz_set_ui(pi, 1);
    mpz_set_ui(theta, 0);
    mpz_set_ui(p, 2);       // Discard prime value 2
    mpz_set_ui(t, 0);

    /* In order to have a sufficient size Emin/Emax, just ensure pi is
     * within 16 bits of bitcnt */
    do {
        mpz_nextprime(p, p);
        mpz_mul(pi, pi, p);
    } while (mpz_sizeinbase(pi, 2) < bitcnt - 16);

    mpz_mul_ui(t, p, 2);
    mpz_clear(p);
}
#endif

static void generator_gd_precompute(mpz_t pi, mpz_t nu, mpz_t lambda, mpz_t rho,
        mp_bitcnt_t bitcnt)
{
    /* This function generates a value, pi, which minimizes the ratio
     * phi(pi) / pi by choosing all prime factors of small powers. Also,
     * we return a value t, which is 2 * p^k, where p^k is the maximum value
     * of any prime power in the value pi. Further, we construct a sequence
     * theta, which represents a bit pattern corresponding to pi's prime
     * factors. */

    mpz_t p, t;
    mpz_init(p);
    mpz_init(t);

    mpz_set_ui(nu, 1);
    mpz_set_ui(lambda, 1);
    mpz_set_ui(p, 2);       // Discard prime value 2

    /* In order to have a sufficient size Emin/Emax, just ensure pi is
     * within 16 bits of bitcnt */
    do {
        mpz_nextprime(p, p);
        mpz_sub_ui(t, p, 1);        // For prime > 2, just use totient value
        mpz_lcm(lambda, lambda, t);
        mpz_mul(nu, nu, p);
    } while (mpz_sizeinbase(nu, 2) < bitcnt - 16);

    /* Find an Emax and Emin to construct rho and pi */
    /* Want Emax * nu = Pi as close to Wmax as possible */

    mpz_mul_ui(pi, nu, 1729);   // These are hardcoded, but alter above loop
                                // to check and have these constructed potentially
    mpz_mul_ui(rho, nu, 4180);

    // Actually set pi

    mpz_clear(p);
    mpz_clear(t);
}

// TODO: IMPLEMENT
#if 0
static void generator_g_compute(mpz_t rop, mpz_t pi, mpz_t lambda,
        gmp_randstate_t rstate, mp_bitcnt_t bitcnt)
{
}
#endif

/* We should call a function which precomputes these values and then
 * pass them to this function ideally to avoid excessive computation */
static void generator_gd_compute(mpz_t rop, mpz_t pi, mpz_t lambda,
        gmp_randstate_t rstate, mp_bitcnt_t bitcnt)
{
    mpz_t t1;
    mpz_init(t1);

    // pi is zero
    mpz_urandomb(rop, rstate, bitcnt);

    do {
        mpz_add_ui(rop, rop, 2);
        mpz_powm(t1, rop, lambda, pi);
    } while (mpz_cmp_ui(t1, 1) != 0);

    mpz_clear(t1);
}

void internal_fast_random_prime(mpz_t rop, gmp_randstate_t rstate, mp_bitcnt_t bitcnt)
{
    /* Wmax = 2^bitcnt - 1
     * Wmin = 2^(bitcnt-1) + 1
     */

    /* We need to find an integer nu, such that minimizing phi(nu)/nu and
     * for which there exist Emin . nu > Wmin and Emax . nu < Wmax - Wmin.
     *
     * Then, Pi = Emax . nu and Rho = Emin . nu
     */

    /* nu must minimize phi(nu) / nu, in other words, we wish to choose
     * only small powers of primes, since phi(p^k) = p^k - p^(k-1). During
     * this generation of pi, we also calculate a small factor, which will
     * be our Emax and in the range [1000, 10000]. As a result, we will
     * be able to calculate the value rho by randomly choosing a value of
     * Emin for which rho is in the required range. */

    mpz_t c, pi, lambda, rho, nu;
    mpz_inits(c, pi, lambda, rho, nu, NULL);

    generator_gd_precompute(pi, nu, lambda, rho, bitcnt);
    generator_gd_compute(c, pi, lambda, rstate, bitcnt);

loop:
    mpz_add(rop, c, rho);
    if (mpz_even_p(rop))
        mpz_add(rop, rop, nu);

    if (mpz_probab_prime_p(rop, 25) == 0) {
        mpz_mul_ui(c, c, 2);
        mpz_mod(c, c, pi);
        goto loop;
    }

    mpz_clears(c, pi, lambda, rho, nu, NULL);
}

/* Generate a random prime of minimum bitcnt number of bits. Currently this
 * doesn't have any other requirements. Strong primes or anything generally
 * are not seen as too useful now, as newer factorization schemes such as the
 * GFNS are not disrupted by this prime choice. */
void internal_naive_random_prime(mpz_t rop, gmp_randstate_t rstate, mp_bitcnt_t bitcnt)
{
    /* Technically in small cases we could get a prime of n + 1 bits */
    mpz_urandomb(rop, rstate, bitcnt);
    mpz_setbit(rop, bitcnt);
    mpz_nextprime(rop, rop);
}

/* Generate a prime rop1 which is equal to 2 * rop2 + 1 where rop2 is also
 * prime */
void internal_naive_random_safe_prime(mpz_t rop1, mpz_t rop2, gmp_randstate_t rstate,
        mp_bitcnt_t bitcnt)
{
    do {
        internal_fast_random_prime(rop2, rstate, bitcnt - 1);
        mpz_mul_ui(rop1, rop2, 2);
        mpz_add_ui(rop1, rop1, 1);
    } while (mpz_probab_prime_p(rop1, 25) == 0);
}

void internal_fast_random_safe_prime(mpz_t rop1, mpz_t rop2, gmp_randstate_t rstate,
        mp_bitcnt_t bitcnt)
{
    /* We can improve this by using a variant which does not redo all the
     * precomputation each prime, and with some other factors. */
    do {
        internal_naive_random_prime(rop1, rstate, bitcnt);
        mpz_sub_ui(rop2, rop1, 1);
        mpz_divexact_ui(rop2, rop2, 2);
    } while (mpz_probab_prime_p(rop2, 25) == 0);
}

/* Chinese remainder theorem case where k = 2 using Bezout's identity. Unlike
 * other mpz functions rop must not be an aliased with any of the other
 * arguments! This is done to save excessive copying in this function, plus
 * it is usually not beneficial as conX_a and conX_m cannot be the same value
 * anyway */
void mpz_2crt(mpz_t rop, mpz_t con1_a, mpz_t con1_m, mpz_t con2_a, mpz_t con2_m)
{
    mpz_t t;
    mpz_init(t);

    mpz_gcd(t, con1_m, con2_m);
    assert(mpz_cmp_ui(t, 1) == 0);

    mpz_invert(rop, con2_m, con1_m);
    mpz_mul(rop, rop, con2_m);
    mpz_mul(rop, rop, con1_a);
    mpz_invert(t, con1_m, con2_m);
    mpz_mul(t, t, con1_m);
    mpz_mul(t, t, con2_a);
    mpz_add(rop, rop, t);
    mpz_mul(t, con1_m, con2_m);
    mpz_mod(rop, rop, t);

    mpz_clear(t);
}

/* Hash an mpz_t value and an unsigned long using ripemd160, and store a
 * corresponding integer into rop. Care is taken to ensure that this is
 * cross-platform and doesn't depend on the order of bytes. */
void mpz_ripemd_mpz_ul(mpz_t rop, mpz_t op1, unsigned long op2) {
    ripemd160_state ctx;
    ripemd160_init(&ctx);

    size_t countp;
    unsigned char *datap = mpz_export(NULL, &countp, 1, 1, -1, 0, op1);
    ripemd160_update(&ctx, datap, countp);
    free(datap);

    // Ripemd160 function will handle different byte ordering for different
    // endian machines.
    const uint64_t op2_u64 = (uint64_t) op2;
    unsigned char *op2_cp = (unsigned char*)&op2_u64;
    ripemd160_update(&ctx, op2_cp, 8);

    unsigned char hdigest[RIPEMD160_DIGEST_SIZE];
    ripemd160_digest(&ctx, hdigest);
    mpz_import(rop, RIPEMD160_DIGEST_SIZE, 1, 1, -1, 0, hdigest);
}

#define HCS_MALLOC_BYTES_REQ(mpz) ((mpz_sizeinbase(mpz, 2) + 7) / 8) * 8

void mpz_ripemd_3mpz_ul(mpz_t rop, mpz_t op1, mpz_t op2, mpz_t op3, unsigned long op4) {
    ripemd160_state ctx;
    ripemd160_init(&ctx);

    size_t countp;
    unsigned char *datap;

    const size_t
        s1 = ((mpz_sizeinbase(op1, 2) + 7) / 8) * 8,
        s2 = ((mpz_sizeinbase(op2, 2) + 7) / 8) * 8,
        s3 = ((mpz_sizeinbase(op3, 2) + 7) / 8) * 8;

    const size_t
    toalloc = s1 > s2
                ? s1 > s3
                    ? s1 : s3
                : s2 > s3
                    ? s2 : s3;

    datap = malloc(toalloc);
    mpz_export(datap, &countp, 1, 1, -1, 0, op1);
    ripemd160_update(&ctx, datap, countp);
    mpz_export(datap, &countp, 1, 1, -1, 0, op2);
    ripemd160_update(&ctx, datap, countp);
    mpz_export(datap, &countp, 1, 1, -1, 0, op3);
    ripemd160_update(&ctx, datap, countp);
    free(datap);

    // Ripemd160 function will handle different byte ordering for different
    // endian machines.
    const uint64_t op4_u64 = (uint64_t) op4;
    unsigned char *op4_cp = (unsigned char*)&op4_u64;
    ripemd160_update(&ctx, op4_cp, 8);

    unsigned char hdigest[RIPEMD160_DIGEST_SIZE];
    ripemd160_digest(&ctx, hdigest);
    mpz_import(rop, RIPEMD160_DIGEST_SIZE, 1, 1, -1, 0, hdigest);
}

#ifdef UTIL_MAIN

#include <time.h>

#define time_process(code)\
    do {\
        clock_t start = clock(), diff;\
        code\
        diff = clock() - start;\
        int msec = diff * 1000 / CLOCKS_PER_SEC;\
        assert(mpz_probab_prime_p(t1, 25) != 0);\
        printf("%ds:%dms ", msec / 1000, msec % 1000);\
    } while (0)

int main(void)
{
    /* Test and compare the speed of safe prime generation. */

    mpz_t t1, t2;
    gmp_randstate_t rstate;
    mpz_inits(t1, t2);
    gmp_randinit_default(rstate);
    mpz_seed(t1, 128);
    gmp_randseed(rstate, t1);

    time_process( internal_naive_random_prime(t1, rstate, 512); );
    time_process( internal_naive_random_prime(t1, rstate, 1024); );
    time_process( internal_naive_random_prime(t1, rstate, 2048); );
    printf("(Naive)\n");

    time_process( internal_fast_random_prime(t1, rstate, 512); );
    time_process( internal_fast_random_prime(t1, rstate, 1024); );
    time_process( internal_fast_random_prime(t1, rstate, 2048); );
    printf("(Fast)\n");

    /*
    printf("NaiveSafe ");
    time_process( internal_naive_random_safe_prime(t1, t2, rstate, 128); );
    time_process( internal_naive_random_safe_prime(t1, t2, rstate, 256); );
    time_process( internal_naive_random_safe_prime(t1, t2, rstate, 512); );
    printf("\n");

    printf("FastSafe: ");
    time_process( internal_fast_random_safe_prime(t1, t2, rstate, 128); );
    time_process( internal_fast_random_safe_prime(t1, t2, rstate, 256); );
    time_process( internal_fast_random_safe_prime(t1, t2, rstate, 512); );
    printf("\n");
    */
}
#endif
