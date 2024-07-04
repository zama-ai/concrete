# Xor distance

We describe how to compute a XOR distance (as known as an Hamming weight distance) in Concrete. This
can be useful in particular for biometry use-cases, where obviously, private is a very interesting
feature.

We present the XOR distance in two contexts, with corresponding codes:
- the XOR distance between [two encrypted tensors](hamming_distance.py)
- the XOR distance between [one encrypted tensor and one clear tensor](hamming_distance_to_clear.py)

Execution times of the different functions are given in the different sections.

## The Goal

Goal here is to compute

```
def dist_in_clear(x, y):
    return numpy.sum(hw(x ^ y))
```

function in FHE. This function XOR to values, and compute `hw`  of the result, which is the Hamming
weight, and is defined as the number of set bits.

For example, for `x = 0xf3 = 0b11110011` and `y = 0xbc = 0x10111100`, we would have
`x ^ y = 0b01001111 = 0x4f`, and so `dist_in_clear(x, y) = 5` since there are 5 set bits in `0x4f`.
Or, in other words, `dist_in_clear(x, y) = 5` means that bits of `x` and bits of `y` differ in 5
different locations:

```
0b11110011
0x10111100
   ^  ^^^^
```

This is a distance function, which can be used for various purpose, including measuring how two
vectors are close to each other. In the context of biometry (or others), it may be very interesting
to compute this function over encrypted `x` and `y` vectors.

## Distance Between Two Encrypted Tensors

### First Implementation

In the [full code](hamming_distance.py), we use a first implementation, which is

```
def dist_in_fhe_directly_from_cp(x, y):
    return np.sum(hw(x ^ y))
```

Here, it's a pure copy of the code in Concrete, and it compiles directly into FHE code!

### Second Implementation with `fhe.bits`

In the [full code](hamming_distance.py), we use a second implementation, which is

```
def dist_in_fhe_with_bits_1b(x, y):
    z = x + y
    zx = fhe.bits(z)[0]
    return np.sum(zx)
```

This function only works for bit-vectors `x` and `y` (as opposed to other functions). Here, we use
`fhe.bits` operator to extract the least-significant bit of the addition `x+y`: indeed, this least
signification bit is exactly `x ^ y`.

### Third Implementation with Concatenation

In the [full code](hamming_distance.py), we use a third implementation, which is

```
def dist_in_fhe_with_xor_internal(x, y, bitsize_w):
    power = 2**bitsize_w
    table = fhe.LookupTable([hw((i % power) ^ (i // power)) for i in range(power**2)])

    z = x + power * y
    zx = table[z]

    return np.sum(zx)
```

Here, we concatenate the elements of `x` and `y` (which are of bitsize `bitsize_w`) into a
`2 * bitsize_w` input, and use a `2 * bitsize_w`-bit programmable bootstrapping.

### Fourth Implementation with `fhe.multivariate`

In the [full code](hamming_distance.py), we use a fourth implementation, which is

```
def dist_in_fhe_with_multivariate_internal(x, y):
    zx = fhe.multivariate(lambda x, y: hw(x ^ y))(x, y)
    return np.sum(zx)
```

Here, we use `fhe.multivariate`, which is a function which takes the two inputs `x` and `y`. Under the hood, it's going to be replaced by a `2 * bitsize_w`-bit programmable bootstrapping.

### Execution Time Between Two Encrypted Tensors

_All of the following timings were measured on an `hpc7a` machine, with Concrete 2.5.1._

If one executes the [code](hamming_distance.py)
for 120-bit vectors (of whatever shape), execution times should be:

```
    dist_in_fhe_with_multivariate_tables on 2 bits:  0.07 seconds
    dist_in_fhe_with_multivariate_tables on 1 bits:  0.09 seconds
             dist_in_fhe_with_xor_tables on 2 bits:  0.09 seconds
            dist_in_fhe_directly_from_cp on 2 bits:  0.10 seconds
             dist_in_fhe_with_xor_tables on 1 bits:  0.11 seconds
            dist_in_fhe_directly_from_cp on 1 bits:  0.12 seconds
                dist_in_fhe_with_bits_1b on 1 bits:  0.15 seconds
    dist_in_fhe_with_multivariate_tables on 3 bits:  0.27 seconds
             dist_in_fhe_with_xor_tables on 3 bits:  0.29 seconds
            dist_in_fhe_directly_from_cp on 3 bits:  0.31 seconds
            dist_in_fhe_directly_from_cp on 4 bits:  1.17 seconds
             dist_in_fhe_with_xor_tables on 4 bits:  2.18 seconds
    dist_in_fhe_with_multivariate_tables on 4 bits:  2.24 seconds

```

For 1200-bit vectors (obtained with `python hamming_distance.py --nb_bits 1200`), execution times
should be:

```
    dist_in_fhe_with_multivariate_tables on 2 bits:  0.22 seconds
             dist_in_fhe_with_xor_tables on 2 bits:  0.29 seconds
            dist_in_fhe_directly_from_cp on 2 bits:  0.32 seconds
            dist_in_fhe_directly_from_cp on 1 bits:  0.36 seconds
             dist_in_fhe_with_xor_tables on 1 bits:  0.39 seconds
                dist_in_fhe_with_bits_1b on 1 bits:  0.44 seconds
    dist_in_fhe_with_multivariate_tables on 1 bits:  0.48 seconds
            dist_in_fhe_directly_from_cp on 3 bits:  0.73 seconds
    dist_in_fhe_with_multivariate_tables on 3 bits:  0.85 seconds
             dist_in_fhe_with_xor_tables on 3 bits:  1.14 seconds
            dist_in_fhe_directly_from_cp on 4 bits:  5.99 seconds
    dist_in_fhe_with_multivariate_tables on 4 bits:  7.17 seconds
             dist_in_fhe_with_xor_tables on 4 bits:  8.20 seconds
```

And finally, for 12804-bit vectors, execution times should be:

```
             dist_in_fhe_with_xor_tables on 2 bits:  2.53 seconds
    dist_in_fhe_with_multivariate_tables on 2 bits:  2.66 seconds
            dist_in_fhe_directly_from_cp on 2 bits:  3.64 seconds
            dist_in_fhe_directly_from_cp on 1 bits:  4.25 seconds
                dist_in_fhe_with_bits_1b on 1 bits:  4.40 seconds
             dist_in_fhe_with_xor_tables on 1 bits:  4.53 seconds
    dist_in_fhe_with_multivariate_tables on 1 bits:  4.71 seconds
            dist_in_fhe_directly_from_cp on 3 bits:  6.76 seconds
    dist_in_fhe_with_multivariate_tables on 3 bits:  7.93 seconds
             dist_in_fhe_with_xor_tables on 3 bits:  8.43 seconds
            dist_in_fhe_directly_from_cp on 4 bits: 23.72 seconds
             dist_in_fhe_with_xor_tables on 4 bits: 39.27 seconds
    dist_in_fhe_with_multivariate_tables on 4 bits: 40.89 seconds
```

## Distance Between One Encrypted Tensor and One Clear Tensor

In [this code](hamming_distance_to_clear.py), we propose a simple implementation for the special case
where one of the vectors (here, `y`) is not encrypted. The function `dist_in_fhe` is based on the
following idea: `x` is seen as a line-vector of bits, while `y` is seen as a column-vector of bits.
`x` and `y` follow a simple transform (before the encryption): bits 0 are mapped to -1, while bits 1
are mapped to 1. Then we just compute the scalar product `u` between mapped `x` and `y`.

Bits which are equal between mapped `x` and `y` will be either (1, 1) or (-1, -1) so corresponding
impact on the sum of the scalar multiplication is a 1. On the opposite, for bits which are different,
so (1, -1) or (-1, 1), the impact on the sum of the scalar multiplication is a -1. All in all,
`u = n - 2 HW(x^y)`, where `n` is the number of bits of `x` (which is the number of bits of `y` too).

In the code, we compute `n - u`, and we divide by 2 after the decryption, which doesn't reduce the
privacy of the computation.

### Execution Time Between One Encrypted Tensor and One Clear Tensor

This case is really fast, since there is no programmable bootstrapping (PBS) in the code. It's a
purely levelled FHE circuit.

For 12804-bit vectors, on an `hpc7a` machine, with Concrete 2.7.0, we have:

```
Computing XOR distance on 12804 bits using algorithm dist_in_fhe, using vectors of 1b cells
Distance between encrypted vectors done in 0.43 seconds in average
```
