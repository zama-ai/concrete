#include "cereal/archives/binary.hpp"
#include "defines.h"
#include "ipcl/ipcl.hpp"
#include "ipcl/pub_key.hpp"
#include <cassert>
#include <cstdint>
#include <ostream>
#include <vector>

template <typename Word>
std::ostream &writeWord(std::ostream &ostream, Word word) {
  ostream.write(reinterpret_cast<char *>(&(word)), sizeof(word));
  assert(ostream.good());
  return ostream;
}

template <typename Word>
std::istream &readWord(std::istream &istream, Word &word) {
  istream.read(reinterpret_cast<char *>(&(word)), sizeof(word));
  assert(istream.good());
  return istream;
}

std::ostream &operator<<(std::ostream &ostream, const PaiPublicKey &key) {
  cereal::BinaryOutputArchive oarchive(ostream);

  oarchive(*key.ptr);

  return ostream;
}
std::ostream &operator<<(std::ostream &ostream, const PaiPrivateKey &key) {
  cereal::BinaryOutputArchive oarchive(ostream);

  oarchive(*key.ptr->getN());
  oarchive(*key.ptr->getP());
  oarchive(*key.ptr->getQ());

  return ostream;
}
std::ostream &operator<<(std::ostream &ostream, const PaiCiphertext &key) {
  cereal::BinaryOutputArchive oarchive(ostream);

  oarchive((const ipcl::PublicKey &)*key.ptr->getPubKey());

  oarchive((const std::vector<BigNumber> &)key.ptr->getTexts());

  return ostream;
}
std::ostream &operator<<(std::ostream &ostream, const BigNumber_ &key) {
  cereal::BinaryOutputArchive oarchive(ostream);

  oarchive(*key.ptr);

  return ostream;
}

std::ostream &operator<<(std::ostream &ostream,
                         const CompressedCiphertext &compCt) {

  ostream << compCt.scale;
  writeWord<uint64_t>(ostream, compCt.paiBitLen);
  writeWord<uint64_t>(ostream, compCt.logScale);
  writeWord<uint64_t>(ostream, compCt.maxCts);

  writeWord<uint64_t>(ostream, compCt.pCts.size());

  for (auto &a : compCt.pCts) {
    ostream << a;
  }

  return ostream;
}

std::istream &operator>>(std::istream &istream, PaiPublicKey &key) {
  cereal::BinaryInputArchive iarchive(istream);

  if (key.ptr != nullptr) {
    delete key.ptr;
  }

  key.ptr = new ipcl::PublicKey();
  iarchive(*key.ptr);

  return istream;
}
std::istream &operator>>(std::istream &istream, PaiPrivateKey &key) {
  cereal::BinaryInputArchive iarchive(istream);

  BigNumber N, P, Q;

  iarchive(N);
  iarchive(P);
  iarchive(Q);

  if (key.ptr != nullptr) {
    delete key.ptr;
  }

  key.ptr = new ipcl::PrivateKey(N, P, Q);

  return istream;
}

std::istream &operator>>(std::istream &istream, PaiCiphertext &key) {

  cereal::BinaryInputArchive iarchive(istream);

  ipcl::PublicKey pub_key;
  std::vector<BigNumber> base_text;

  iarchive(pub_key);
  iarchive(base_text);

  if (key.ptr != nullptr) {
    delete key.ptr;
  }

  key.ptr = new ipcl::CipherText(pub_key, base_text);

  return istream;
}
std::istream &operator>>(std::istream &istream, BigNumber_ &key) {
  cereal::BinaryInputArchive iarchive(istream);

  if (key.ptr != nullptr) {
    delete key.ptr;
  }

  key.ptr = new BigNumber();

  iarchive(*key.ptr);

  return istream;
}

std::istream &operator>>(std::istream &istream, CompressedCiphertext &compCt) {

  istream >> compCt.scale;
  readWord<uint64_t>(istream, compCt.paiBitLen);
  readWord<uint64_t>(istream, compCt.logScale);
  readWord<uint64_t>(istream, compCt.maxCts);

  uint64_t size;
  readWord<uint64_t>(istream, size);

  compCt.pCts.clear();
  for (uint64_t i = 0; i < size; i++) {

    auto a = PaiCiphertext();

    istream >> a;

    compCt.pCts.push_back(std::move(a));
  }

  return istream;
}
