#include "compress_lwe/defines.h"
#include "libhcs++/djcs.hpp"
#include <cmath>
#include <cstdint>
#include <gmpxx.h>
#include <iostream>
#include <libhcs++.hpp>
#include <memory>
#include <utility>

namespace comp {

mpz::~mpz() {
  if (ptr != nullptr) {
    delete (mpz_class *)ptr;
  }
}

mpz::mpz(const mpz &other) { ptr = new mpz_class(*(mpz_class *)other.ptr); }

mpz &mpz::operator=(const mpz &other) {
  if (ptr != nullptr) {
    delete (mpz_class *)ptr;
  }
  ptr = new mpz_class(*(mpz_class *)other.ptr);
  return *this;
}

PublicKey::~PublicKey() {
  if (ptr != nullptr) {
    delete ptr;
  }
}
PrivateKey::~PrivateKey() {
  if (ptr != nullptr) {
    delete ptr;
  }
}

CompressedCiphertext::CompressedCiphertext()
    : lwe_dim(0), maxCts(0), ahe_cts() {
  scale.ptr = new mpz_class(0);

  auto random = std::make_shared<hcs::random>();

  ahe_pk = std::make_shared<PublicKey>(new hcs::djcs::public_key(random));
}

CompressedCiphertext::CompressedCiphertext(std::shared_ptr<PublicKey> &pk,
                                           uint64_t lwe_dim)
    : ahe_pk(pk), lwe_dim(lwe_dim) {
  uint64_t bitwidth, s;
  s = ahe_pk->ptr->as_ptr()->s;

  bitwidth = 64;

  double ahe_capacity = 2047.0 * s;
  uint64_t logScale = (uint64_t)ceil(log2(lwe_dim + 1) + bitwidth);

  mpz_class raw_scale = 1;
  raw_scale <<= logScale;
  scale.ptr = new mpz_class(raw_scale);
  maxCts = (uint64_t)std::floor(ahe_capacity / (double)logScale);
}
} // namespace comp
