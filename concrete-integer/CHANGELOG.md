# 0.1.1

## Fixed

  - Fix builds on aarch64 due to missing concrete-core feature activation
  - Bump to rust edition 2021

---

# 0.1.0

## Added

  - Crt functions (sub, neg, scalar_add, scalar_sub, scalar_mul)
  - Parallelized Crt functions (`_parallelized` function of `ServerKey`)
  - Parallelized Radix functions (`_parallelized` function of `ServerKey`)
  - Initial support for aarch64 (requires nightly)
  - Look Up Table (LUT) generation and evaluation via WoP-PBS for all supported representations

## Changed

  - Replaced fftw with concrete-fft
  - Improved API (**Breaking changes**):
    * Split the `Ciphertext` struct into two structs `RadixCiphertext` and `CrtCiphertex`.
    * Added `RadixClientKey` and `CrtClientKey` (The more general `ClientKey` still exist)
    
## Removed

 - TreePBS related functions and structures.