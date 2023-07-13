#include "compress_lwe/serialize.h"
#include "compress_lwe/defines.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <gmpxx.h>
#include <iostream>
#include <libhcs++.hpp>
#include <string>
#include <vector>

namespace comp {

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

std::ostream &writeString(std::ostream &ostream, std::string &str) {
  writeWord<uint64_t>(ostream, str.size());

  for (char &a : str) {
    writeWord<char>(ostream, a);
  }

  return ostream;
}

std::istream &readString(std::istream &istream, std::string &str) {
  uint64_t size;

  readWord<uint64_t>(istream, size);

  str.clear();

  for (int i = 0; i < size; i++) {
    char a;

    readWord<char>(istream, a);

    str.push_back(a);
  }

  return istream;
}

std::ostream &write_mpz(std::ostream &ostream, const mpz_class &value) {
  long int length;

  mpz_get_d_2exp(&length, value.get_mpz_t());

  double d_length = length;

  uint32_t block_length = std::ceil(d_length / 32.);

  writeWord<uint32_t>(ostream, block_length);

  mpz_class remainder = value;

  for (uint32_t i = 0; i < block_length; i++) {
    mpz_class msb = remainder >> 32;

    mpz_class lsb = remainder - (msb << 32);

    auto lsb2 = lsb.get_mpz_t();

    uint32_t lsb_raw = mpz_get_ui(lsb2);

    writeWord<uint32_t>(ostream, lsb_raw);

    remainder = msb;
  }

  if (remainder != 0) {
    exit(-1);
  }

  assert(ostream.good());
  return ostream;
}

std::istream &read_mpz(std::istream &istream, mpz_class &value) {

  uint32_t block_length;

  readWord<uint32_t>(istream, block_length);

  value = 0;

  for (uint32_t i = 0; i < block_length; i++) {
    uint32_t part;
    readWord<uint32_t>(istream, part);

    mpz_class part_mpz = part;

    value += part_mpz << (32 * i);
  }

  assert(istream.good());
  return istream;
}

std::ostream &operator<<(std::ostream &ostream, const PublicKey &key) {
  std::string json = key.ptr->export_json();

  writeString(ostream, json);

  return ostream;
}
std::ostream &operator<<(std::ostream &ostream, const PrivateKey &key) {
  std::string json = key.ptr->export_json();

  writeString(ostream, json);
  return ostream;
}
std::ostream &operator<<(std::ostream &ostream, const mpz &key) {
  write_mpz(ostream, *(mpz_class *)key.ptr);

  return ostream;
}
std::ostream &operator<<(std::ostream &ostream, const mpz_vec &key) {
  writeWord<uint64_t>(ostream, key.size());
  for (auto &a : key) {
    ostream << a;
  }

  return ostream;
}

std::ostream &writeCompCt(std::ostream &ostream,
                          const CompressedCiphertext &compCt) {

  ostream << compCt.scale;
  ostream << compCt.ahe_cts;
  ostream << *compCt.ahe_pk;
  writeWord<uint64_t>(ostream, compCt.lwe_dim);
  writeWord<uint64_t>(ostream, compCt.maxCts);

  return ostream;
}

std::ostream &writeFullKeys(std::ostream &ostream, const FullKeys &keys) {
  ostream << *keys.ahe_pk;
  ostream << *keys.ahe_sk;
  ostream << keys.compKey;

  return ostream;
}

std::ostream &writeCompKeys(std::ostream &ostream,
                            const comp::CompressionKey &keys) {
  ostream << *keys.ahe_pk;
  ostream << keys.compKey;

  return ostream;
}

std::istream &operator>>(std::istream &istream, PublicKey &key) {
  if (key.ptr == nullptr) {
    auto random = std::make_shared<hcs::random>();
    key.ptr = new hcs::djcs::public_key(random);
  }

  std::string json;

  readString(istream, json);

  key.ptr->import_json(json);

  return istream;
}
std::istream &operator>>(std::istream &istream, PrivateKey &key) {

  if (key.ptr == nullptr) {

    auto random = std::make_shared<hcs::random>();
    key.ptr = new hcs::djcs::private_key(random);
  }

  std::string json;

  readString(istream, json);

  key.ptr->import_json(json);

  return istream;
}
std::istream &operator>>(std::istream &istream, mpz &key) {
  if (key.ptr == nullptr) {
    key.ptr = new mpz_class();
  }

  read_mpz(istream, *(mpz_class *)key.ptr);

  return istream;
}

std::istream &operator>>(std::istream &istream, mpz_vec &key) {
  uint64_t size;
  readWord<uint64_t>(istream, size);
  key.clear();
  for (uint64_t i = 0; i < size; i++) {
    mpz a;

    istream >> a;

    key.push_back(std::move(a));
  }
  return istream;
}

std::istream &readCompCt(std::istream &istream, CompressedCiphertext &compCt) {

  istream >> compCt.scale;
  istream >> compCt.ahe_cts;
  istream >> *compCt.ahe_pk;
  readWord<uint64_t>(istream, compCt.lwe_dim);
  readWord<uint64_t>(istream, compCt.maxCts);

  return istream;
}

std::istream &readFullKeys(std::istream &istream, FullKeys &keys) {
  auto random = hcs::random();

  keys.ahe_pk = std::make_shared<comp::PublicKey>();
  keys.ahe_sk = std::make_shared<comp::PrivateKey>();

  istream >> *keys.ahe_pk;

  istream >> *keys.ahe_sk;
  istream >> keys.compKey;

  return istream;
}

std::istream &readCompKeys(std::istream &istream, CompressionKey &keys) {
  auto random = hcs::random();

  keys.ahe_pk = std::make_shared<comp::PublicKey>();

  istream >> *keys.ahe_pk;
  istream >> keys.compKey;

  return istream;
}

} // namespace comp
