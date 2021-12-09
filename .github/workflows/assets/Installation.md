# Installation

You can either install the compiler in user space or globally (you need root/sudo access):

1. User space install: extract the tarball to a chosen path and add chosen/path/zamacompiler/bin to your $PATH.

2. Global install: extract the tarball to a temporary path , and copy

- temporary/path/zamacompiler/bin/zamacompiler to /usr/bin (or a directory in $PATH)
- temporary/path/zamacompiler/lib/libZamalangRuntime.so to /usr/lib (or another lib folder)