# SHA1 computation with Modules

This document demonstrates the use of Modules in Concrete through a SHA1 computation example. SHA1
is a deprecated and broken hash function; we use it here for pedagogical purposes, not for its
security.

The SHA1 code is available [here](sha1.py). Execution times for the different functions are provided
in the final section. We created our example by forking
[python-sha1](https://github.com/ajalt/python-sha1) by [AJ Alt](https://github.com/ajalt) and made
extensive modifications to implement SHA1 in FHE and corresponding tests.

## SHA1 overview

SHA1 is a deprecated broken hash function defined by NIST in 1995: You can find detailed information
on SHA1 on its [Wikipedia page](https://en.wikipedia.org/wiki/SHA-1) or in its
[official description](https://nvlpubs.nist.gov/nistpubs/fips/nist.fips.180-4.pdf). We follow the
structure of its [pseudo-code](https://en.wikipedia.org/wiki/SHA-1#SHA-1_pseudocode) in our
implementation.

## Our FHE implementation

In our implementation, only the compression function is implemented in FHE, corresponding to the
`_process_encrypted_chunk_server_side function`. The rest is done client-side in the clear,
including the message expansion. While more of the process could be done in FHE, this tutorial
focuses on demonstrating the use of [Modules](https://docs.zama.ai/concrete/compilation/combining/composing_functions_with_modules).

Our Module contains 7 functions which can be combined together:
- `xor3` XORs three values together
- `iftern` computes the ternary IF, i.e., c ? t : f
- `maj` computes the majority function
- `rotate30` rotates a 32-bit word by 30 positions
- `rotate5` rotates a 32-bit word by 5 positions
- `add2` adds two values together
- `add5` adds five values together


```
@fhe.module()
class MyModule:
    @fhe.function({"x": "encrypted", "y": "encrypted", "z": "encrypted"})
    def xor3(x, y, z):
        return x ^ y ^ z

    @fhe.function({"x": "encrypted", "y": "encrypted", "z": "encrypted"})
    def iftern(x, y, z):
        return z ^ (x & (y ^ z))

    @fhe.function({"x": "encrypted", "y": "encrypted", "z": "encrypted"})
    def maj(x, y, z):
        return (x & y) | (z & (x | y))

    @fhe.function({"x": "encrypted"})
    def rotate30(x):
        ans = fhe.zeros((32,))
        ans[30:32] = x[0:2]
        ans[0:30] = x[2:32]
        return ans

    @fhe.function({"x": "encrypted"})
    def rotate5(x):
        ans = fhe.zeros((32,))
        ans[5:32] = x[0:27]
        ans[0:5] = x[27:32]
        return ans

    @fhe.function({"x": "encrypted", "y": "encrypted"})
    def add2(x, y):
        return fhe.bits(add_chunked_number(x, y))[0]

    @fhe.function(
        {"x": "encrypted", "y": "encrypted", "u": "encrypted", "v": "encrypted", "w": "encrypted"}
    )
    def add5(x, y, u, v, w):
        result = add_chunked_number(x, y)
        result = add_chunked_number(result, u)
        result = add_chunked_number(result, v)
        result = add_chunked_number(result, w)

        return fhe.bits(result)[0]
```

We then compile this Module, setting `p_error=10**-8` as a very small value to avoid computation
errors. The Module feature allows the combination of all these functions so that the outputs of
some to be used as inputs for others. This makes it convenient to create larger functions with
some control flow (conditions, branches, loops) handled in the clear while using these smaller
functions. In our case, this is done in the `_process_encrypted_chunk_server_side function.`

## Details of `_process_encrypted_chunk_server_side`

`_process_encrypted_chunk_server_side` uses encrypted inputs and returns encrypted values. In the
clear, all variables are 32-bit words, but here they are represented as 32 encrypted bits to
simplify and accelerate the non-linear operations in SHA1.

Then, we have the main loop of the compression function:

```
    for i in range(80):
        if 0 <= i <= 19:

            # Do f = d ^ (b & (c ^ d))
            fsplit_enc = my_module.iftern.run(bsplit_enc, csplit_enc, dsplit_enc)

            ksplit = split(0x5A827999)
        elif 20 <= i <= 39:

            # Do f = b ^ c ^ d
            fsplit_enc = my_module.xor3.run(bsplit_enc, csplit_enc, dsplit_enc)

            ksplit = split(0x6ED9EBA1)
        elif 40 <= i <= 59:

            # Do f = (b & c) | (b & d) | (c & d)
            fsplit_enc = my_module.maj.run(bsplit_enc, csplit_enc, dsplit_enc)

            ksplit = split(0x8F1BBCDC)
        elif 60 <= i <= 79:

            # Do f = b ^ c ^ d
            fsplit_enc = my_module.xor3.run(bsplit_enc, csplit_enc, dsplit_enc)

            ksplit = split(0xCA62C1D6)
```

In this main loop, we take the right choice of `f` and `k`. Here, we can see a first use of
the different functions in the Module.

Then we continue with other functions in the Module, to compute `arot5 = _left_rotate(a, 5)` and
`arot5 + f + e + k + w[i]`:

```
        # Do arot5 = _left_rotate(a, 5)
        arot5split_enc = my_module.rotate5.run(asplit_enc)

        # Do arot5 + f + e + k + w[i]
        ssplit_enc = my_module.add5.run(
            arot5split_enc,
            fsplit_enc,
            esplit_enc,
            wsplit_enc[i],
            my_module.rotate5.encrypt(ksplit),  # BCM: later remove the encryption on k
        )
```

Finally, we update the different `a, b, c, d, e` values as in the clear
implementation but with the encrypted forms:

```
        # Final update of the a, b, c, d and e registers
        newasplit_enc = ssplit_enc

        esplit_enc = dsplit_enc
        dsplit_enc = csplit_enc

        # Do c = _left_rotate(b, 30)
        csplit_enc = my_module.rotate30.run(bsplit_enc)

        bsplit_enc = asplit_enc
        asplit_enc = newasplit_enc
```

You can see that we compiled the Module's different functions on inputset made with
bits. Under the hood, Concrete adds a few programmable bootstrappings to compute the correct
functions in FHE.


## MLIR code

Compiling with `show_mlir = True` allows to see the different MLIR implementations.

## Testing or using

This tutorial focuses on the use of Modules rather than a production-ready implementation. For a
full client-server API, you might want to perform more operations in FHE, including message
expansion, and function optimizations.

You can verify the implementation in FHE by running `python sha1.py --autotest`: it will
pick a certain number of random inputs, hash them in FHE and compare the result with the `hashlib`
standard implementation.

You can also hash a given value with
`echo -n "The quick brown fox jumps over the lazy dog" | python sha1.py`, and it will print
something like:

```
sha1-digest: 2fd4e1c67a2d28fced849ee1bb76e7391b93eb12
computed in: 320.265383 seconds
```

## Benchmarks

We have executed our implementation on an HPC7a machine with Concrete 2.7.

`python sha1.py --autotest` typically returns:

```
Checking SHA1(yozulCBAPuFosqTBMwPTVmvQvmfhGFJjdtSSiemdytn) for an input length 43
sha1-digest: aa1871c2d560221e14f18b43d559aafc4920d9bc
computed in: 93.844131 seconds
Checking SHA1(HwXFZxXUGckiuWysDtrpIijiRwRGPJZPGaNpJMlfbPptfNhzKOXZMiZnoLlaRCXqK) for an input length 65
sha1-digest: 8555e05fc2396a3b291e983901bdfa02cb454a72
computed in: 181.812173 seconds
Checking SHA1(am) for an input length 2
sha1-digest: 96e8155732e8324ae26f64d4516eb6fe696ac84f
computed in: 91.206739 seconds
Checking SHA1(OTzaWtYfzqKyTHIgBSlmI) for an input length 21
sha1-digest: c76426dbecb3afd015b132e0e44a1b4d0fc664cb
computed in: 91.182635 seconds
Checking SHA1(MBwAxKkvLOUzXkHILdVchwjfcUTlofyQdSqaonqcXvRVVwEJpmaGKOsNDCUGkt) for an input length 62
sha1-digest: 2231cecd803a9000c117a2f2d3ea35f70ecf6fce
computed in: 179.300580 seconds
Checking SHA1(iqObpuNHZXKztrUZYrpmAPjFflNLacYyUTBLZdbjbPcjbLOseIKqZNYCbsoaDuwvgbvfWE) for an input length 70
sha1-digest: 87cdf7a7f984ef5843d0cfc95a69eaef3e82e31b
computed in: 182.332336 seconds
Checking SHA1(nPIZWXYUXOerncJAeBQrcPhuHsXbYcQkKMRoAGzxFZjBXWvproNcHlHIFNNGzQChEVjZvsEmSOpQuoihPgudlqizwAfXzgU) for an input length 95
sha1-digest: 601db1e338e8b817f5309c7c3bf89450d70425a1
computed in: 182.233933 seconds
Checking SHA1(SfUDDjhLqcmifCpLlqnUZKFjwtPjfCrpdRChUWypdrhdDTjizF) for an input length 50
sha1-digest: 8c57e70e92af96c6df5aecb4f3740e68b0686883
computed in: 91.378782 seconds
Checking SHA1(VePVJXHljlLpkThrANZaEkkbYoFZVdSFFsvVkQPDlrwyisOIZqAUhGwHYYhxnFjOrUgFV) for an input length 69
sha1-digest: b54a98dc0f74b50669a5e0c81199a2819e26d3e9
computed in: 182.419465 seconds
Checking SHA1(hCOuPvLrBXYhaWLeSnyRxpJZmXyEeTCBdfkgvDwvFIgoVaXNPzHoHkitSvZLosTMEcDpuoLGqeCkuwLmQQLbSibizfPwxp) for an input length 94
sha1-digest: 19432e37fe0bbc7c5a5c9379ed38c4298d796fe6
computed in: 182.285430 seconds
Checking SHA1(IRZuLcbBXJAfgYMuFwGWUXiKgfoWwSAOTOoQiLaZHKwuTbGGsTChMLEdlaRXkIOYUFJRxRyAVHhQbFKyKOFDNairicMm) for an input length 92
sha1-digest: 4307a80365937a3ff6703763a8f1a759362c35e5
computed in: 182.162872 seconds
Checking SHA1(hsTWqDAwjdLiAhmLsnBJJozejydFLcksQmSQVcojeAhdZizXMzjKSAGjQUjRPGZlwGaKnE) for an input length 70
sha1-digest: 3889e4448d11e9d801608ab3a94378ee8d41aaa0
computed in: 182.288777 seconds
Checking SHA1(GihWwpolRavMdWHiNGVlOVsziPqZcYkPEyTmgfbBtwMowfKMipRxADcqqzMBlpCrWEOJpnZQcqAwt) for an input length 77
sha1-digest: 4b048588c8ec5d1602487124b5232889668f6b6a
computed in: 182.184901 seconds
Checking SHA1(GiSbjYHQyEozGliG) for an input length 16
sha1-digest: dc7a31c2e17ae503c6c808a02c05f8d1ad4b346f
computed in: 91.260812 seconds
Checking SHA1(boVGLdzvbKOTYUSErbeSyoiJQMox) for an input length 28
sha1-digest: 50a40fb6d0362087ed9b6dceda5378c14b96a743
computed in: 91.119592 seconds
Checking SHA1(kiaAjFJrzBaRFvSgIzIQdlJZokXGPBpjNRPqEcVDyCFGkxNzeiRNHuSmD) for an input length 57
sha1-digest: 44e423eaa7616d2a93b235d99b61f76a6f58e236
computed in: 179.129039 seconds
Checking SHA1(IylnzdjqldIJGkolzRlJhFz) for an input length 23
sha1-digest: 463137090865b283801d49d1546ce0dea45f2529
computed in: 91.211171 seconds
Checking SHA1(QEyszfMlKJjKIf) for an input length 14
sha1-digest: 937bfdff3bb97e9037a23bca4055302f97a90eb3
computed in: 91.124824 seconds
Checking SHA1(IOTanBIVaq) for an input length 10
sha1-digest: b8dca6ad5df447549f5a95d4c195b889f5be6069
computed in: 91.110348 seconds
Checking SHA1(FiWWroKJVGeMPfmaNKmXdalAyIRLpYJdFgCfBIEMXPDfWR) for an input length 46
sha1-digest: 84775d2c31df2ab191a0417cf26028f989d2cd9c
computed in: 91.031444 seconds
Checking SHA1()
sha1-digest: da39a3ee5e6b4b0d3255bfef95601890afd80709
computed in: 91.044046 seconds
Checking SHA1(The quick brown fox jumps over the lazy dog)
sha1-digest: 2fd4e1c67a2d28fced849ee1bb76e7391b93eb12
computed in: 90.971023 seconds
```

These results mean that:
- one block of compression takes about 92 seconds
- two blocks of compression take about 181 seconds
