
#include "ipcl/ipcl.hpp"
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "defines.h"

PaiPublicKey::~PaiPublicKey() {
  if (ptr != nullptr)
    delete ptr;
}

PaiPrivateKey::~PaiPrivateKey() {
  if (ptr != nullptr)
    delete ptr;
}

PaiCiphertext::~PaiCiphertext() {
  if (ptr != nullptr) {
    delete ptr;
  }
}
PaiCiphertext::PaiCiphertext(const PaiCiphertext &other) {
  ptr = new ipcl::CipherText(*other.ptr);
}

PaiCiphertext &PaiCiphertext::operator=(const PaiCiphertext &other) {
  if (ptr != nullptr) {
    delete ptr;
  }
  ptr = new ipcl::CipherText(*other.ptr);
  return *this;
}

BigNumber_::~BigNumber_() {
  if (ptr != nullptr)
    delete ptr;
}

LWEParams::LWEParams(uint64_t n, uint64_t log_q)
    : qBig(BigNumber_(new BigNumber(1))), n(n), logQ(log_q) {

  for (uint64_t i = 0; i < log_q; i++) {
    *qBig.ptr *= 2;
  }
}

CompressedCiphertext::CompressedCiphertext(uint64_t n, uint64_t log_q)
    : scale(BigNumber_(new BigNumber(1))) {
  logScale = (uint64_t)ceil(log2(n + 1) + 2 * log_q);

  for (uint64_t i = 0; i < logScale; i++) {
    *scale.ptr *= 2;
  }
  maxCts = (uint64_t)std::floor((float)paiBitLen / logScale);
}
