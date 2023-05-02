#ifndef COMPRESSLWE_SERIALIZATION_H
#define COMPRESSLWE_SERIALIZATION_H

#include "defines.h"

std::ostream &operator<<(std::ostream &ostream, const PaiPublicKey &key);
std::ostream &operator<<(std::ostream &ostream, const PaiPrivateKey &key);
std::ostream &operator<<(std::ostream &ostream, const PaiCiphertext &key);
std::ostream &operator<<(std::ostream &ostream, const BigNumber_ &key);
std::ostream &operator<<(std::ostream &ostream,
                         const CompressedCiphertext &compCt);

std::istream &operator>>(std::istream &istream, PaiPublicKey &key);
std::istream &operator>>(std::istream &istream, PaiPrivateKey &key);
std::istream &operator>>(std::istream &istream, PaiCiphertext &key);
std::istream &operator>>(std::istream &istream, BigNumber_ &key);
std::istream &operator>>(std::istream &istream, CompressedCiphertext &compCt);

#endif // COMPRESSLWE_SERIALIZATION_H
