# CompresssLWE
CompressLWE is an open-source library, built on top of pailliercryptolib, that allows for compression of LWE ciphertexts in transit from server to client in an efficient way

## Building the Library

### Prerequisites
For best performance, especially due to the multi-buffer modular exponentiation function, the library is to be used on AVX512IFMA enabled systems, as listed below in Intel CPU codenames:
- Intel Cannon Lake
- Intel Ice Lake
- Intel Sapphire Rapids

The library can be built and used without AVX512IFMA and/or QAT, if the features are not supported. But for better performance, it is recommended to use the library on Intel Xeon® scalable processors - Ice Lake-SP or Sapphire Rapids-SP Xeon CPUs while fully utilizing the features.

The following operating systems have been tested and deemed to be fully functional.
- Ubuntu 18.04 and higher
- Red Hat Enterprise Linux 8.1 and higher
- CentOS Stream

We will keep working on adding more supported operating systems.
### Dependencies
Must have dependencies include:
```
cmake >= 3.15.1
git
pthread
Intel C++ Compiler Classic 2021.3 for Linux* OS
Intel oneAPI DPC++/C++ Compiler for Linux* OS >= 2021.3
g++ >= 8.0
clang >= 10.0
```

The following libraries and tools are also required,
```
nasm >= 2.15
OpenSSL >= 1.1.0
```

```OpenSSL``` can be installed with:
```bash
# Ubuntu
sudo apt install libssl-dev
# Fedora (RHEL 8, Centos)
sudo dnf install openssl-devel
```

In order to install ```nasm```, please refer to the [Netwide Assembler webpage](https://nasm.us/) for download and installation details.

### Instructions
for building IPCL, clone their [repo](https://github.com/intel/pailliercryptolib), then:
```
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/path/to/install/ -DCMAKE_BUILD_TYPE=Release -DIPCL_TEST=OFF -DIPCL_BENCHMARK=OFF
cmake --build build -j
cmake --build build --target install
```
For building compressLWE:
```
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=[PATH TO IPCL]
make
```
# Contributors
Main contributors to this project, sorted by alphabetical order of first name are:
- [Abdulrahman Diaa](https://www.linkedin.com/in/abdulrahman-diaa-555300126/)
- [Rasoul Akhavan Mahdavi](https://rasoulam.github.io/)
