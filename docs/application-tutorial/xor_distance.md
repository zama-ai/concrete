# Xor Distance

We describe how to compute a XOR distance (as known as an Hamming weight distance) in Concrete. This
can be useful in particular for biometry use-cases, where obviously, private is a very interesting
feature.

The full code can be done [here](../../frontends/concrete-python/examples/xor_distance/hamming_distance.py). Execution times of the different functions are given in the
final section.

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

## First Implementation

In the [full code](../../frontends/concrete-python/examples/xor_distance/hamming_distance.py), we use a first implementation, which is

```
def dist_in_fhe_directly_from_cp(x, y):
    return np.sum(hw(x ^ y))
```

Here, it's a pure copy of the code in Concrete, and it compiles directly into FHE code!

## Second Implementation with `fhe.bits`

In the [full code](../../frontends/concrete-python/examples/xor_distance/hamming_distance.py), we use a second implementation, which is

```
def dist_in_fhe_with_bits_1b(x, y):
    z = x + y
    zx = fhe.bits(z)[0]
    return np.sum(zx)
```

This function only works for bit-vectors `x` and `y` (as opposed to other functions). Here, we use
`fhe.bits` operator to extract the least-significant bit of the addition `x+y`: indeed, this least
signification bit is exactly `x ^ y`.

## Third Implementation with Concatenation

In the [full code](../../frontends/concrete-python/examples/xor_distance/hamming_distance.py), we use a third implementation, which is

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

## Fourth Implementation with `fhe.multivariate`

In the [full code](../../frontends/concrete-python/examples/xor_distance/hamming_distance.py), we use a fourth implementation, which is

```
def dist_in_fhe_with_multivariate_internal(x, y):
    zx = fhe.multivariate(lambda x, y: hw(x ^ y))(x, y)
    return np.sum(zx)
```

Here, we use `fhe.multivariate`, which is a function which takes the two inputs `x` and `y`. Under the hood, it's going to be replaced by a `2 * bitsize_w`-bit programmable bootstrapping.

## Execution Time

_All of the following timings were measured on an `hpc7a` machine, with Concrete 2.5.1._

If one executes the [code](../../frontends/concrete-python/examples/xor_distance/hamming_distance.py)
for 120-bit vectors (of whatever shape), execution times should be:

```
    dist_in_fhe_with_multivariate_tables on 2 bits:  0.07 secunds
    dist_in_fhe_with_multivariate_tables on 1 bits:  0.09 secunds
             dist_in_fhe_with_xor_tables on 2 bits:  0.09 secunds
            dist_in_fhe_directly_from_cp on 2 bits:  0.10 secunds
             dist_in_fhe_with_xor_tables on 1 bits:  0.11 secunds
            dist_in_fhe_directly_from_cp on 1 bits:  0.12 secunds
                dist_in_fhe_with_bits_1b on 1 bits:  0.15 secunds
    dist_in_fhe_with_multivariate_tables on 3 bits:  0.27 secunds
             dist_in_fhe_with_xor_tables on 3 bits:  0.29 secunds
            dist_in_fhe_directly_from_cp on 3 bits:  0.31 secunds
            dist_in_fhe_directly_from_cp on 4 bits:  1.17 secunds
             dist_in_fhe_with_xor_tables on 4 bits:  2.18 secunds
    dist_in_fhe_with_multivariate_tables on 4 bits:  2.24 secunds

```

For 1200-bit vectors (obtained with `python hamming_distance.py --nb_bits 1200`), execution times
should be:

```
    dist_in_fhe_with_multivariate_tables on 2 bits:  0.22 secunds
             dist_in_fhe_with_xor_tables on 2 bits:  0.29 secunds
            dist_in_fhe_directly_from_cp on 2 bits:  0.32 secunds
            dist_in_fhe_directly_from_cp on 1 bits:  0.36 secunds
             dist_in_fhe_with_xor_tables on 1 bits:  0.39 secunds
                dist_in_fhe_with_bits_1b on 1 bits:  0.44 secunds
    dist_in_fhe_with_multivariate_tables on 1 bits:  0.48 secunds
            dist_in_fhe_directly_from_cp on 3 bits:  0.73 secunds
    dist_in_fhe_with_multivariate_tables on 3 bits:  0.85 secunds
             dist_in_fhe_with_xor_tables on 3 bits:  1.14 secunds
            dist_in_fhe_directly_from_cp on 4 bits:  5.99 secunds
    dist_in_fhe_with_multivariate_tables on 4 bits:  7.17 secunds
             dist_in_fhe_with_xor_tables on 4 bits:  8.20 secunds
```

And finally, for 12804-bit vectors, execution times should be:

```
             dist_in_fhe_with_xor_tables on 2 bits:  2.53 secunds
    dist_in_fhe_with_multivariate_tables on 2 bits:  2.66 secunds
            dist_in_fhe_directly_from_cp on 2 bits:  3.64 secunds
            dist_in_fhe_directly_from_cp on 1 bits:  4.25 secunds
                dist_in_fhe_with_bits_1b on 1 bits:  4.40 secunds
             dist_in_fhe_with_xor_tables on 1 bits:  4.53 secunds
    dist_in_fhe_with_multivariate_tables on 1 bits:  4.71 secunds
            dist_in_fhe_directly_from_cp on 3 bits:  6.76 secunds
    dist_in_fhe_with_multivariate_tables on 3 bits:  7.93 secunds
             dist_in_fhe_with_xor_tables on 3 bits:  8.43 secunds
            dist_in_fhe_directly_from_cp on 4 bits: 23.72 secunds
             dist_in_fhe_with_xor_tables on 4 bits: 39.27 secunds
    dist_in_fhe_with_multivariate_tables on 4 bits: 40.89 secunds
```

