#ifndef COMPRESSLWE_DEFINES_H
#define COMPRESSLWE_DEFINES_H

#include <memory>
#include <vector>

struct BigNumber;

namespace ipcl {
struct PublicKey;
struct PrivateKey;
struct CipherText;
} // namespace ipcl

class PaiPublicKey {
public:
  PaiPublicKey() : ptr(nullptr){};
  PaiPublicKey(ipcl::PublicKey *ptr) : ptr(ptr){};
  PaiPublicKey(const PaiPublicKey &) = delete;
  PaiPublicKey(PaiPublicKey &&other) : ptr(other.ptr) { other.ptr = nullptr; };
  ~PaiPublicKey();

  ipcl::PublicKey *ptr;
};

class PaiPrivateKey {
public:
  PaiPrivateKey() : ptr(nullptr){};
  PaiPrivateKey(ipcl::PrivateKey *ptr) : ptr(ptr){};
  PaiPrivateKey(const PaiPrivateKey &) = delete;
  PaiPrivateKey(PaiPrivateKey &&other) : ptr(other.ptr) {
    other.ptr = nullptr;
  };
  ~PaiPrivateKey();

  ipcl::PrivateKey *ptr;
};
class PaiCiphertext {
public:
  PaiCiphertext() : ptr(nullptr){};
  PaiCiphertext(ipcl::CipherText *ptr) : ptr(ptr){};
  PaiCiphertext(const PaiCiphertext &other);
  PaiCiphertext(PaiCiphertext &&other) : ptr(other.ptr) {
    other.ptr = nullptr;
  };
  ~PaiCiphertext();
  PaiCiphertext &operator=(const PaiCiphertext &other);

  ipcl::CipherText *ptr;
};

struct BigNumber_ {
public:
  BigNumber_() : ptr(nullptr){};
  BigNumber_(BigNumber *ptr) : ptr(ptr){};
  BigNumber_(const BigNumber_ &) = delete;
  BigNumber_(BigNumber_ &&other) : ptr(other.ptr) { other.ptr = nullptr; };
  ~BigNumber_();

  BigNumber *ptr;
};

struct PaiFullKeys {
  std::shared_ptr<PaiPublicKey> pub_key;
  std::shared_ptr<PaiPrivateKey> priv_key;
  std::shared_ptr<PaiCiphertext> compKey; // A vector of paillier ciphers
};

struct LWEParams {
  uint64_t n; // Ciphertext dimension of underlying lwe scheme
  uint64_t logQ;
  BigNumber_ qBig;

  LWEParams(uint64_t n, uint64_t log_q);
};

struct CompressedCiphertext {
  BigNumber_ scale; // width of every lwe cipher in packed paillier cipher
  uint64_t paiBitLen = 2048;

  uint64_t logScale;
  uint64_t maxCts;

  std::vector<PaiCiphertext> pCts;

  CompressedCiphertext(uint64_t n, uint64_t log_q);
};

#endif // COMPRESSLWE_DEFINES_H
