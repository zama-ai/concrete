# CompresssLWE
CompressLWE is an open-source library that allows for compression of LWE ciphertexts in transit from server to client in an efficient way.

## Building the Library

### Dependencies
Must have dependencies include:
```
cmake >= 3.22
g++
GMP
```
The following command can install GMP and cmake
```
sudo apt-get install libgmp-dev cmake
```

### Installing modified libhcs
We also require a modified verison of [libhcs](https://github.com/tiehuis/libhcs), which we have include the source code in the ```libhcs``` directory. Please refer to that directory to install the modified version of libhcs.

### Building CompressLWE
For building compressLWE, you can run the following commands in the root directory of the repository:
```
mkdir build
cd build
cmake ..
make
```

## Examples
An example of how to use the library is provided in ```main.cpp```.


# Contributors
Main contributors to this project, sorted by alphabetical order of first name are:
- [Abdulrahman Diaa](https://www.linkedin.com/in/abdulrahman-diaa-555300126/)
- [Rasoul Akhavan Mahdavi](https://rasoulam.github.io/)
