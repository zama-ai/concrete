# 0.1.1

## Fixed

- Bump to rust edition 2021

---

# 0.1.0

## Added

  - Initial support for aarch64 (requires nightly)
  - Look Up Table (LUT) generation and evaluation via WoP-PBS for all supported representations

## Changed

  - Replaced fftw with concrete-fft
  - Replaced concrete-core-experimental dependency with concrete-core 1.0.0
  
## Removed

 - TreePBS related functions and structures.