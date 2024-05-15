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
focuses on demonstrating the use of [Modules](https://docs.zama.ai/concrete/compilation/modules).

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
        ans = fhe.zeros((32,))
        cy = 0

        for i in range(32):
            t = x[i] + y[i] + cy
            cy, tr = t >= 2, t % 2
            ans[i] = tr

        return ans

    @fhe.function(
        {"x": "encrypted", "y": "encrypted", "u": "encrypted", "v": "encrypted", "w": "encrypted"}
    )
    def add5(x, y, u, v, w):
        ans = fhe.zeros((32,))
        cy = 0

        for i in range(32):
            t = x[i] + y[i] + cy
            cy, tr = t // 2, t % 2
            ans[i] = tr

        cy = 0

        for i in range(32):
            t = ans[i] + u[i] + cy
            cy, tr = t // 2, t % 2
            ans[i] = tr

        cy = 0

        for i in range(32):
            t = ans[i] + v[i] + cy
            cy, tr = t // 2, t % 2
            ans[i] = tr

        cy = 0

        for i in range(32):
            t = ans[i] + w[i] + cy
            cy, tr = t // 2, t % 2
            ans[i] = tr

        return ans
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

We have executed our implementation on an HPC7a machine with Concrete 2.7.0rc1.

`python sha1.py --autotest` typically returns:

```
Checking SHA1(fdASguUMBwhPcKuDpPqoRlQXLrLQbnxEvPJSQSIUDTBoaqrJlBualgoWEINmDZDYSuGuSOpGBWwWzjAfktWYZZUliv) for an input length 90
sha1-digest: 5bb539fd423875ccc8a33148dae724f5b2cf9391
computed in: 295.306287 seconds
Checking SHA1(BYwXTbqE) for an input length 8
sha1-digest: 90a8dcad6ddff7ca8fd487b80a37fcd250c56bed
computed in: 145.341164 seconds
Checking SHA1(rnPZh) for an input length 5
sha1-digest: 47610d2c26ee8b45ab0f4c8f8e4d405b2cd37f1f
computed in: 145.318081 seconds
Checking SHA1(orRaJMGbUJtxITQvqiOCPjKJWYuHomuiexCQQgZyTeAAFJcgCftDCRAkcLKjRECelIMPQphGEUlSNthE) for an input length 80
sha1-digest: bd74b4e64349d308f3b95b54cf61ee416bdd6b18
computed in: 288.240576 seconds
Checking SHA1(ROokDcdczajNPjlCPoWotaRJHBtOVyiyxMIIeCtxaDCjk) for an input length 45
sha1-digest: 1ff546c3a64f27339781c095cbc097f392c2cccd
computed in: 143.621941 seconds
Checking SHA1(KbCXFt) for an input length 6
sha1-digest: 7e5789f0c83fa5102004fbeeef3ac22244d1cdac
computed in: 143.509567 seconds
Checking SHA1(mpKnkHtrgokxgQSzcIjFtxKnhmMfZbIbkJavnkSxW) for an input length 41
sha1-digest: 1308d9f7cba634ab2617edb5116b8bdf434f16f5
computed in: 143.341450 seconds
Checking SHA1(oauoWKJGyjjTcXqRIxFGuVuMwiwjKYfttQ) for an input length 34
sha1-digest: 60367153b7049ca92eb979ad7b809c5a3f47a64e
computed in: 143.693254 seconds
Checking SHA1(ZMGiaIOmBJPncOsUCxj) for an input length 19
sha1-digest: fafba9f2fe6b5a0fddad4ad765909c8fc32117c6
computed in: 143.720215 seconds
Checking SHA1(HwCXIHnFoGUgIBqaQrrpDnhEvPBX) for an input length 28
sha1-digest: 5224cace20f8d20fa3ea8d9974b5ff3a0be7fd48
computed in: 143.523006 seconds
Checking SHA1(AfyzsimngrqeWoqZKOBRwVuvttfgJTpegMbiHjUNdWzTg) for an input length 45
sha1-digest: 8ca27aca1c362ca63e50d58aa7065b4322f028a0
computed in: 143.481069 seconds
Checking SHA1(hNEUPakrqQpGGZvtHvht) for an input length 20
sha1-digest: 36ae34ed85e62ac0f922e36fc98b23e725695be1
computed in: 143.478666 seconds
Checking SHA1(CjgfYYlNKqZdHeXFfqTwhycbGBeSpzpxKPwWItriiNKZCcEJRZlM) for an input length 52
sha1-digest: 3c012f41c5fe4581f80e2901fc4bbbb70ff7a9ba
computed in: 143.490262 seconds
Checking SHA1(EXIGkYzWpcqpfRKCSbBJJqqmUBkFwWfPGooJvsVAshWjMr) for an input length 46
sha1-digest: 2518c4d13ec7608f59632ac993b726e572c3aaae
computed in: 143.840785 seconds
Checking SHA1(sgzaAqZnhXmFJOJMyfGxweYFMmLeUHmMCWETfqzstzpFYKaGpnasiLHPTcJtukHztEQpXzquREcbtoJDaoqjfM) for an input length 86
sha1-digest: 46f4b0653ed7ea0ce89cc18f6720e5e334d63a45
computed in: 288.155301 seconds
Checking SHA1(oRaisdHJovDxCnwyComEGejqMceBTOVhJucVnwgC) for an input length 40
sha1-digest: 909f9c6275aa9f41d8ecaf52203bb0e24cf978d7
computed in: 143.466817 seconds
Checking SHA1(mtTWxtHerQgLdBGftWdiCwBKqtu) for an input length 27
sha1-digest: 624a7dcec460061a2a6499dae978fe4afd674110
computed in: 145.389956 seconds
Checking SHA1(beYzkJLvZMmoXbQwqoVThpyaQ) for an input length 25
sha1-digest: 25a9df47bd055384a9ee614c1dc7213c04f2087c
computed in: 147.234881 seconds
Checking SHA1(CpQWXXRNlXIoSZNxmXUwWHqmUAdlOrDyZPzzOhznlpGntrUgvktlZ) for an input length 53
sha1-digest: f2bde6574d8f6aa360929f6a5f919700b16e093b
computed in: 147.154393 seconds
Checking SHA1(busWigrVdsXnkjTh) for an input length 16
sha1-digest: fe47568d433278a38a4729f7891d03eaacdb0e40
computed in: 147.465694 seconds
Checking SHA1()
sha1-digest: da39a3ee5e6b4b0d3255bfef95601890afd80709
computed in: 147.256297 seconds
Checking SHA1(The quick brown fox jumps over the lazy dog)
sha1-digest: 2fd4e1c67a2d28fced849ee1bb76e7391b93eb12
computed in: 147.697102 seconds
```

These results mean that:
- one block of compression takes about 147 seconds
- two blocks of compression take about 290 seconds
