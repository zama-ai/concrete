#include <iostream>
#include <cassert>
#include <chrono>
#include "library.h"

using namespace std::chrono;

void test_single_lwe_compress() {

    uint64_t n = 630;
    uint64_t log_q = 64;
    uint64_t p = 32;
    bool binaryKeys = false;
    size_t s = 1;

    // Generate a random vector<uint64_t> of size n with values in [0, q) and call it the secret key
    std::vector<uint64_t> lwe_key(n);
    for (uint64_t i = 0; i < n; i++) {
        if (binaryKeys){
            lwe_key[i] = sample(1);
        } else {
            lwe_key[i] = sample(log_q);
        }
    }

    // Generate a ciphertext
    std::vector<uint64_t> lwe_ct(n + 1);
    for (uint64_t i = 0; i < n + 1; i++) {
        lwe_ct[i] = sample(log_q);
    }

    // Call the generate Keys function with the lwe_key
    Keys* key = generateKeys(lwe_key, s);

    // Call the compressSingle function with the compression key, the ciphertext, and the LWEParams
    LWEParams params(n, log_q, p, binaryKeys);
    mpz_class compressed_ct = compressSingle(key->compKey, key->ahe_pk, lwe_ct, params);

    // Call the decryptCompressedSingle function with the compressed ciphertext, the paillier private key, and the LWEParams
    mpz_class decrypted = decryptCompressedSingle(compressed_ct, key->ahe_sk, params);

    mpz_class lwe_decrypted = decryptLWE(lwe_ct, lwe_key, params);
    gmp_printf("%Zd <--> %Zd\n", lwe_decrypted.get_mpz_t(), decrypted.get_mpz_t());

    assert(decrypted == lwe_decrypted);

    // print success message
    std::cout << "Single Test Passed!" << std::endl;

}

void test_batched_compress(uint64_t num_cts, uint64_t s) {
    uint64_t n = 630;
    uint64_t log_q = 64;
    uint64_t p = 32;
    bool binaryKeys = false;

    assert(log_q <= 64);

    // Get starting timepoint
    system_clock::time_point start = high_resolution_clock::now();
    // Generate a lwe key of size n with values in [0, q)
    std::vector<uint64_t> lwe_key(n);
    for (uint64_t i = 0; i < n; i++) {
        if (binaryKeys){
            lwe_key[i] = sample(1);
        } else {
            lwe_key[i] = sample(log_q);
        }
    }
    // Get ending timepoint
    system_clock::time_point stop = high_resolution_clock::now();
    // Get duration. Substart timepoints to
    microseconds d_lwe_key = duration_cast<microseconds>(stop - start);

    start=stop;

    // Generate a batch of lwe ciphertexts
    std::vector<std::vector<uint64_t>> lwe_cts(num_cts);
    for (uint64_t i = 0; i < num_cts; i++) {
        lwe_cts[i] = std::vector<uint64_t>(n + 1);
        for (uint64_t j = 0; j < n + 1; j++) {
            lwe_cts[i][j] = sample(log_q);
        }
    }
    // Get ending timepoint
    stop = high_resolution_clock::now();
    // Get duration. Substart timepoints to
    microseconds d_lwe_cts = duration_cast<microseconds>(stop - start);

    start = stop;

    Keys* key = generateKeys(lwe_key, s);

    // Get ending timepoint
    stop = high_resolution_clock::now();
    // Get duration. Substart timepoints to
    microseconds d_keygen = duration_cast<microseconds>(stop - start);

    start=stop;

    LWEParams params(n, log_q, p, binaryKeys);

    CompressedCiphertext compressed_ct = compressBatched(
            key->compKey,
            key->ahe_pk,
            lwe_cts,
            params
    );

    // Get ending timepoint
    stop = high_resolution_clock::now();
    // Get duration. Substart timepoints to
    microseconds d_compress = duration_cast<microseconds>(stop - start);

    start = stop;

    mpz_vec decrypted = decryptCompressedBatched(
            compressed_ct,
            key->ahe_sk,
            params,
            num_cts,
            true
    );

    // Get ending timepoint
    stop = high_resolution_clock::now();
    // Get duration. Substart timepoints to
    microseconds d_decrypt = duration_cast<microseconds>(stop - start);

    bool test=true;
    for (uint64_t i = 0; i < num_cts; i++) {
        mpz_class lwe_decrypted = decryptLWE(lwe_cts[i], lwe_key, params);
//        gmp_printf("%Zd <--> %Zd\n", lwe_decrypted.get_mpz_t(), decrypted[i].get_mpz_t());
        if (decrypted[i] != lwe_decrypted){
            test=false;
        }
    }
    assert(test);
    if (test){
        std::cout << "Encrypt-first Test Passed!" << std::endl;
        std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>"<< std::endl;
        std::cout << "S: " <<s<<" ; Ciphers:  "<< num_cts<< std::endl;
        std::cout << "Number of Damgard CTs:  " << compressed_ct.ahe_cts.size()<< " ciphers"<< std::endl;
        std::cout << "Time taken by compress: " << d_compress.count()/1000 << " ms" << std::endl;

        std::cout << "Time taken by LWE key:  " << d_lwe_key.count() << " microseconds" << std::endl;
        std::cout << "Time taken by LWE cts:  " << d_lwe_cts.count() << " microseconds" << std::endl;
        std::cout << "Time taken by keygen:   " << d_keygen.count() << " microseconds" << std::endl;
        std::cout << "Time taken by decrypt:  " << d_decrypt.count() << " microseconds" << std::endl;
        std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>"<< std::endl;

    } else {
        std::cout << "Test Failed!" << std::endl;
    }
    
}


//void test_batched_lwe_compress(uint64_t num_cts, uint64_t s) {
//    uint64_t n = 630;
//    uint64_t log_q = 64;
//    uint64_t p = 32;
//
//    assert(log_q <= 64);
//
//    // Get starting timepoint
//    system_clock::time_point start = high_resolution_clock::now();
//    // Generate a lwe key of size n with values in [0, q)
//    std::vector<uint64_t> lwe_key(n);
//    for (uint64_t i = 0; i < n; i++) {
//        lwe_key[i] = sample(log_q);
//    }
//    // Get ending timepoint
//    system_clock::time_point stop = high_resolution_clock::now();
//    // Get duration. Substart timepoints to
//    microseconds d_lwe_key = duration_cast<microseconds>(stop - start);
//
//    start=stop;
//
//    // Generate a batch of lwe ciphertexts
//    std::vector<std::vector<uint64_t>> lwe_cts(num_cts);
//    for (uint64_t i = 0; i < num_cts; i++) {
//        lwe_cts[i] = std::vector<uint64_t>(n + 1);
//        for (uint64_t j = 0; j < n + 1; j++) {
//            lwe_cts[i][j] = sample(log_q);
//        }
//    }
//    // Get ending timepoint
//    stop = high_resolution_clock::now();
//    // Get duration. Substart timepoints to
//    microseconds d_lwe_cts = duration_cast<microseconds>(stop - start);
//
//    start = stop;
//
//    Keys* key = generateKeys(lwe_key, s);
//
//    // Get ending timepoint
//    stop = high_resolution_clock::now();
//    // Get duration. Substart timepoints to
//    microseconds d_keygen = duration_cast<microseconds>(stop - start);
//
//    start=stop;
//
//    LWEParams params(n, log_q, p);
//
//    CompressedCiphertext compressed_ct = compressBatchedScaleFirst(
//            key->compKey,
//            key->ahe_pk,
//            lwe_cts,
//            params
//    );
//
//    // Get ending timepoint
//    stop = high_resolution_clock::now();
//    // Get duration. Substart timepoints to
//    microseconds d_compress = duration_cast<microseconds>(stop - start);
//
//    start = stop;
//
//    mpz_vec decrypted = decryptCompressedBatched(
//            compressed_ct,
//            key->ahe_sk,
//            params,
//            num_cts,
//            true
//    );
//
//    // Get ending timepoint
//    stop = high_resolution_clock::now();
//    // Get duration. Substart timepoints to
//    microseconds d_decrypt = duration_cast<microseconds>(stop - start);
//
//    for (uint64_t i = 0; i < num_cts; i++) {
//        mpz_class lwe_decrypted = decryptLWE(lwe_cts[i], lwe_key, params);
////        gmp_printf("%Zd <--> %Zd\n", lwe_decrypted.get_mpz_t(), decrypted[i].get_mpz_t());
//        assert(decrypted[i] == lwe_decrypted);
//    }
//    std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>"<< std::endl;
//    std::cout << "Scale-first Test Passed!" << std::endl;
//    std::cout << "S: " <<s<<" ; Ciphers:  "<< num_cts<< std::endl;
//    std::cout << "Number of Damgard CTs:  " << compressed_ct.ahe_cts.size()<< " ciphers"<< std::endl;
//    std::cout << "Time taken by compress: " << d_compress.count()/1000 << " ms" << std::endl;
//
//    std::cout << "Time taken by LWE key:  " << d_lwe_key.count() << " microseconds" << std::endl;
//    std::cout << "Time taken by LWE cts:  " << d_lwe_cts.count() << " microseconds" << std::endl;
//    std::cout << "Time taken by keygen:   " << d_keygen.count() << " microseconds" << std::endl;
//    std::cout << "Time taken by decrypt:  " << d_decrypt.count() << " microseconds" << std::endl;
//    std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>"<< std::endl;
//}
//


int main() {
    srand(time(nullptr));
    test_single_lwe_compress();
    uint64_t num_cts = 784;
    for (size_t s = 1; s < 8; ++s) {
        test_batched_compress(num_cts, s);
//        test_batched_lwe_compress(num_cts, s);
    }
    return 0;
}