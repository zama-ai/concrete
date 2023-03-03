# Installation

You can either install the compiler in user space or globally (you need root/sudo access):

1. User space install: extract the tarball to a chosen path and make the lib, bin, and include directories accessible depending on your needs.

2. Global install: extract the tarball to a temporary path , and copy

- temporary/path/concretecompiler/bin/* inside /usr/local/bin/ (or a directory in $PATH)
- temporary/path/concretecompiler/lib/* inside /usr/local/lib/ (or another lib folder)
- temporary/path/concretecompiler/include/* inside /usr/local/include/ (or another include folder)
