from .nd import NoiseDistribution, stddevf
from .lwe_parameters import LWEParameters

#
# Kyber
#
#
# https://pq-crystals.org/kyber/data/kyber-specification-round3-20210804.pdf
# Table 1, Page 11, we are ignoring the compression
#
# https://eprint.iacr.org/2020/1308.pdf
# Table 2, page 27, disagrees on Kyber 512

Kyber512 = LWEParameters(
    n=2 * 256,
    q=3329,
    Xs=NoiseDistribution.CenteredBinomial(3),
    Xe=NoiseDistribution.CenteredBinomial(3),
    m=2 * 256,
    tag="Kyber 512",
)

Kyber768 = LWEParameters(
    n=3 * 256,
    q=3329,
    Xs=NoiseDistribution.CenteredBinomial(2),
    Xe=NoiseDistribution.CenteredBinomial(2),
    m=3 * 256,
    tag="Kyber 768",
)

Kyber1024 = LWEParameters(
    n=4 * 256,
    q=3329,
    Xs=NoiseDistribution.CenteredBinomial(2),
    Xe=NoiseDistribution.CenteredBinomial(2),
    m=4 * 256,
    tag="Kyber 1024",
)

#
# Saber
#
#
# https://www.esat.kuleuven.be/cosic/pqcrypto/saber/files/saberspecround3.pdf
# Table 1, page 11
#
# https://eprint.iacr.org/2020/1308.pdf
# Table 2, page 27, agrees

LightSaber = LWEParameters(
    n=2 * 256,
    q=8192,
    Xs=NoiseDistribution.CenteredBinomial(5),
    Xe=NoiseDistribution.UniformMod(7),
    m=2 * 256,
    tag="LightSaber",
)

Saber = LWEParameters(
    n=3 * 256,
    q=8192,
    Xs=NoiseDistribution.CenteredBinomial(4),
    Xe=NoiseDistribution.UniformMod(7),
    m=3 * 256,
    tag="Saber",
)

FireSaber = LWEParameters(
    n=4 * 256,
    q=8192,
    Xs=NoiseDistribution.CenteredBinomial(3),
    Xe=NoiseDistribution.UniformMod(7),
    m=4 * 256,
    tag="FireSaber",
)

NTRUHPS2048509Enc = LWEParameters(
    n=508,
    q=2048,
    Xe=NoiseDistribution.SparseTernary(508, 2048 / 16 - 1),
    Xs=NoiseDistribution.UniformMod(3),
    m=508,
    tag="NTRUHPS2048509Enc",
)

NTRUHPS2048677Enc = LWEParameters(
    n=676,
    q=2048,
    Xs=NoiseDistribution.UniformMod(3),
    Xe=NoiseDistribution.SparseTernary(676, 2048 / 16 - 1),
    m=676,
    tag="NTRUHPS2048677Enc",
)

NTRUHPS4096821Enc = LWEParameters(
    n=820,
    q=4096,
    Xs=NoiseDistribution.UniformMod(3),
    Xe=NoiseDistribution.SparseTernary(820, 4096 / 16 - 1),
    m=820,
    tag="NTRUHPS4096821Enc",
)

NTRUHRSS701Enc = LWEParameters(
    n=700,
    q=8192,
    Xs=NoiseDistribution.UniformMod(3),
    Xe=NoiseDistribution.UniformMod(3),
    m=700,
    tag="NTRUHRSS701",
)

NISTPQC_R3 = (
    Kyber512,
    Kyber768,
    Kyber1024,
    LightSaber,
    Saber,
    FireSaber,
    NTRUHPS2048509Enc,
    NTRUHPS2048677Enc,
    NTRUHPS4096821Enc,
    NTRUHRSS701Enc,
)

HESv111024128error = LWEParameters(
    n=1024,
    q=2 ** 27,
    Xs=NoiseDistribution.DiscreteGaussian(3.0),
    Xe=NoiseDistribution.DiscreteGaussian(3.0),
    m=1024,
    tag="HESv11error",
)

HESv111024128ternary = LWEParameters(
    n=1024,
    q=2 ** 27,
    Xs=NoiseDistribution.UniformMod(3),
    Xe=NoiseDistribution.DiscreteGaussian(3.0),
    m=1024,
    tag="HESv11ternary",
)

HESv11 = (HESv111024128error, HESv111024128ternary)


# FHE schemes

# TFHE
# https://tfhe.github.io/tfhe/security_and_params.html
TFHE630 = LWEParameters(
    n=630,
    q=2 ** 32,
    Xs=NoiseDistribution.UniformMod(2),
    Xe=NoiseDistribution.DiscreteGaussian(stddev=2 ** (-15) * 2 ** 32),
    tag="TFHE630",
)

TFHE1024 = LWEParameters(
    n=1024,
    q=2 ** 32,
    Xs=NoiseDistribution.UniformMod(2),
    Xe=NoiseDistribution.DiscreteGaussian(stddev=2 ** (-25) * 2 ** 32),
    tag="TFHE630",
)

# https://eprint.iacr.org/2018/421.pdf
# Table 3, page 55
TFHE16_500 = LWEParameters(
    n=500,
    q=2 ** 32,
    Xs=NoiseDistribution.UniformMod(2),
    Xe=NoiseDistribution.DiscreteGaussian(stddev=2.43 * 10 ** (-5) * 2 ** 32),
    tag="TFHE16_500",
)

TFHE16_1024 = LWEParameters(
    n=1024,
    q=2 ** 32,
    Xs=NoiseDistribution.UniformMod(2),
    Xe=NoiseDistribution.DiscreteGaussian(stddev=3.73 * 10 ** (-9) * 2 ** 32),
    tag="TFHE16_1024",
)

# https://eprint.iacr.org/2018/421.pdf
# Table 4, page 55
TFHE20_612 = LWEParameters(
    n=612,
    q=2 ** 32,
    Xs=NoiseDistribution.UniformMod(2),
    Xe=NoiseDistribution.DiscreteGaussian(stddev=2 ** (-15) * 2 ** 32),
    tag="TFHE20_612",
)

TFHE20_1024 = LWEParameters(
    n=1024,
    q=2 ** 32,
    Xs=NoiseDistribution.UniformMod(2),
    Xe=NoiseDistribution.DiscreteGaussian(stddev=2 ** (-26) * 2 ** 32),
    tag="TFHE20_1024",
)

# FHEW
# https://eprint.iacr.org/2014/816.pdf
# page 14
FHEW = LWEParameters(
    n=500,
    q=2 ** 32,
    Xs=NoiseDistribution.UniformMod(2),
    Xe=NoiseDistribution.DiscreteGaussian(stddev=2 ** (-15) * 2 ** 32),
    tag="FHEW",
)

# SEAL

# v2.0
# https://www.microsoft.com/en-us/research/wp-content/uploads/2016/09/sealmanual.pdf
# Table 3, page 19

SEAL20_1024 = LWEParameters(
    n=1024,
    q=2 ** 48 - 2 ** 20 + 1,
    Xs=NoiseDistribution.UniformMod(3),
    Xe=NoiseDistribution.DiscreteGaussian(stddev=3.19),
    tag="SEAL20_1024",
)

SEAL20_2048 = LWEParameters(
    n=2048,
    q=2 ** 94 - 2 ** 20 + 1,
    Xs=NoiseDistribution.UniformMod(3),
    Xe=NoiseDistribution.DiscreteGaussian(stddev=3.19),
    tag="SEAL20_2048",
)

SEAL20_4096 = LWEParameters(
    n=4096,
    q=2 ** 190 - 2 ** 30 + 1,
    Xs=NoiseDistribution.UniformMod(3),
    Xe=NoiseDistribution.DiscreteGaussian(stddev=3.19),
    tag="SEAL20_4096",
)

SEAL20_8192 = LWEParameters(
    n=8192,
    q=2 ** 383 - 2 ** 33 + 1,
    Xs=NoiseDistribution.UniformMod(3),
    Xe=NoiseDistribution.DiscreteGaussian(stddev=3.19),
    tag="SEAL20_8192",
)

SEAL20_16384 = LWEParameters(
    n=16384,
    q=2 ** 767 - 2 ** 56 + 1,
    Xs=NoiseDistribution.UniformMod(3),
    Xe=NoiseDistribution.DiscreteGaussian(stddev=3.19),
    tag="SEAL20_16384",
)

# v2.2
# https://www.microsoft.com/en-us/research/wp-content/uploads/2017/06/sealmanual_v2.2.pdf
# Table 3, page 20
SEAL22_2048 = LWEParameters(
    n=2048,
    q=2 ** 60 - 2 ** 14 + 1,
    Xs=NoiseDistribution.UniformMod(3),
    Xe=NoiseDistribution.DiscreteGaussian(stddev=3.19),
    tag="SEAL22_2048",
)

SEAL22_4096 = LWEParameters(
    n=4096,
    q=2 ** 116 - 2 ** 18 + 1,
    Xs=NoiseDistribution.UniformMod(3),
    Xe=NoiseDistribution.DiscreteGaussian(stddev=3.19),
    tag="SEAL22_4096",
)

SEAL22_8192 = LWEParameters(
    n=8192,
    q=2 ** 226 - 2 ** 26 + 1,
    Xs=NoiseDistribution.UniformMod(3),
    Xe=NoiseDistribution.DiscreteGaussian(stddev=3.19),
    tag="SEAL22_8192",
)

SEAL22_16384 = LWEParameters(
    n=16384,
    q=2 ** 435 - 2 ** 33 + 1,
    Xs=NoiseDistribution.UniformMod(3),
    Xe=NoiseDistribution.DiscreteGaussian(stddev=3.19),
    tag="SEAL22_16384",
)

SEAL22_32768 = LWEParameters(
    n=32768,
    q=2 ** 889 - 2 ** 54 - 2 ** 53 - 2 ** 52 + 1,
    Xs=NoiseDistribution.UniformMod(3),
    Xe=NoiseDistribution.DiscreteGaussian(stddev=3.19),
    tag="SEAL22_32768",
)

# The following are not parameters of actual schemes
# but useful for benchmarking

# HElib
# https://eprint.iacr.org/2017/047.pdf
# Table 1, page 6
# 80-bit security
HElib80_1024 = LWEParameters(
    n=1024,
    q=2 ** 47,
    Xs=NoiseDistribution.SparseTernary(n=1024, p=32),
    Xe=NoiseDistribution.DiscreteGaussian(stddev=3.2),
    tag="HElib80_1024",
)

HElib80_2048 = LWEParameters(
    n=2048,
    q=2 ** 87,
    Xs=NoiseDistribution.SparseTernary(n=2048, p=32),
    Xe=NoiseDistribution.DiscreteGaussian(stddev=3.2),
    tag="HElib80_2048",
)

HElib80_4096 = LWEParameters(
    n=4096,
    q=2 ** 167,
    Xs=NoiseDistribution.SparseTernary(n=4096, p=32),
    Xe=NoiseDistribution.DiscreteGaussian(stddev=3.2),
    tag="HElib80_4096",
)

# 120-bit security
HElib120_1024 = LWEParameters(
    n=1024,
    q=2 ** 38,
    Xs=NoiseDistribution.SparseTernary(n=1024, p=32),
    Xe=NoiseDistribution.DiscreteGaussian(stddev=3.2),
    tag="HElib80_1024",
)

HElib120_2048 = LWEParameters(
    n=2048,
    q=2 ** 70,
    Xs=NoiseDistribution.SparseTernary(n=2048, p=32),
    Xe=NoiseDistribution.DiscreteGaussian(stddev=3.2),
    tag="HElib80_2048",
)

HElib120_4096 = LWEParameters(
    n=4096,
    q=2 ** 134,
    Xs=NoiseDistribution.SparseTernary(n=4096, p=32),
    Xe=NoiseDistribution.DiscreteGaussian(stddev=3.2),
    tag="HElib80_4096",
)


# Test parameters from CHHS
# https://eprint.iacr.org/2019/1114.pdf
# Table 4, page 18
CHHS_1024_25 = LWEParameters(
    n=1024,
    q=2 ** 25,
    Xs=NoiseDistribution.SparseTernary(n=1024, p=32),
    Xe=NoiseDistribution.DiscreteGaussian(stddev=stddevf(8)),
    tag="CHHS_1024_25",
)

CHHS_2048_38 = LWEParameters(
    n=2048,
    q=2 ** 38,
    Xs=NoiseDistribution.SparseTernary(n=2048, p=32),
    Xe=NoiseDistribution.DiscreteGaussian(stddev=stddevf(8)),
    tag="CHHS_2048_38",
)

CHHS_2048_45 = LWEParameters(
    n=2048,
    q=2 ** 45,
    Xs=NoiseDistribution.SparseTernary(n=2048, p=32),
    Xe=NoiseDistribution.DiscreteGaussian(stddev=stddevf(8)),
    tag="CHHS_2048_45",
)

CHHS_4096_67 = LWEParameters(
    n=4096,
    q=2 ** 67,
    Xs=NoiseDistribution.SparseTernary(n=4096, p=32),
    Xe=NoiseDistribution.DiscreteGaussian(stddev=stddevf(8)),
    tag="CHHS_4096_67",
)

CHHS_4096_82 = LWEParameters(
    n=4096,
    q=2 ** 82,
    Xs=NoiseDistribution.SparseTernary(n=4096, p=32),
    Xe=NoiseDistribution.DiscreteGaussian(stddev=stddevf(8)),
    tag="CHHS_4096_82",
)
