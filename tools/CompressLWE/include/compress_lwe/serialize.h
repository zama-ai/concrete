#ifndef COMPRESSLWE_SERIALIZE_H
#define COMPRESSLWE_SERIALIZE_H

#include "defines.h"

namespace comp {

std::ostream &operator<<(std::ostream &ostream, const comp::PublicKey &key);
std::ostream &operator<<(std::ostream &ostream, const comp::PrivateKey &key);
std::ostream &operator<<(std::ostream &ostream, const comp::mpz &key);
std::ostream &operator<<(std::ostream &ostream, const comp::mpz_vec &key);
std::ostream &writeCompCt(std::ostream &ostream,
                          const comp::CompressedCiphertext &compCt);
std::ostream &writeFullKeys(std::ostream &ostream, const comp::FullKeys &keys);
std::ostream &writeCompKeys(std::ostream &ostream,
                            const comp::CompressionKey &keys);

std::istream &operator>>(std::istream &istream, comp::PublicKey &key);
std::istream &operator>>(std::istream &istream, comp::PrivateKey &key);
std::istream &operator>>(std::istream &istream, comp::mpz &key);
std::istream &operator>>(std::istream &istream, comp::mpz_vec &key);
std::istream &readCompCt(std::istream &istream,
                         comp::CompressedCiphertext &compCt);
std::istream &readFullKeys(std::istream &istream, comp::FullKeys &keys);
std::istream &readCompKeys(std::istream &istream, comp::CompressionKey &keys);

} // namespace comp

#endif // COMPRESSLWE_SERIALIZE_H
