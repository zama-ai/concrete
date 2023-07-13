#ifndef COMPRESSLWE_DEFINES_H
#define COMPRESSLWE_DEFINES_H

#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace hcs {

struct random;

namespace djcs {

struct public_key;
struct private_key;

} // namespace djcs
} // namespace hcs

namespace comp {

class PublicKey {
public:
  PublicKey() : ptr(nullptr){};
  PublicKey(hcs::djcs::public_key *ptr) : ptr(ptr){};
  PublicKey(const PublicKey &);
  PublicKey(PublicKey &&other) : ptr(other.ptr) { other.ptr = nullptr; };
  ~PublicKey();

  hcs::djcs::public_key *ptr;
};

class PrivateKey {
public:
  PrivateKey() : ptr(nullptr){};
  PrivateKey(hcs::djcs::private_key *ptr) : ptr(ptr){};
  // PrivateKey(const PrivateKey &);
  PrivateKey(PrivateKey &&other) : ptr(other.ptr) { other.ptr = nullptr; };
  ~PrivateKey();

  hcs::djcs::private_key *ptr;
};

class mpz {
public:
  mpz() : ptr(nullptr){};
  mpz(void *ptr) : ptr(ptr){};
  mpz(const mpz &);
  mpz &operator=(const mpz &);
  mpz(mpz &&other) : ptr(other.ptr) { other.ptr = nullptr; };
  ~mpz();

  void *ptr;
};

typedef std::vector<mpz> mpz_vec;

struct CompressionKey {
  mpz_vec compKey;
  std::shared_ptr<PublicKey> ahe_pk;
};

struct FullKeys {
  std::shared_ptr<PublicKey> ahe_pk;
  std::shared_ptr<PrivateKey> ahe_sk;
  mpz_vec compKey; // A vector of paillier ciphers

  CompressionKey compression_key() { return CompressionKey{compKey, ahe_pk}; }
};

struct CompressedCiphertext {
  mpz scale;
  mpz_vec ahe_cts;
  std::shared_ptr<PublicKey> ahe_pk;
  uint64_t lwe_dim;
  uint64_t maxCts;

  CompressedCiphertext();
  CompressedCiphertext(std::shared_ptr<PublicKey> &pk, uint64_t lwe_dim);
};
} // namespace comp

#endif // COMPRESSLWE_DEFINES_H
