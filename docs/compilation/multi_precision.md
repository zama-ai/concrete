# Multi Precision

This document explains the multi-precision option for bit-width assignment for integers. 

The multi-precision option enables the frontend to use the smallest bit-width possible for each operation in Fully Homomorphic Encryption (FHE), improving computation efficiency.

## Bit-width and encoding differences
Each integer in the circuit has a certain bit-width, which is determined by the input-set. These bit-widths are visible when graphs are printed, for example:

```
%0 = x                  # EncryptedScalar<uint3>              ∈ [0, 7]
%1 = y                  # EncryptedScalar<uint4>              ∈ [0, 15]
%2 = add(%0, %1)        # EncryptedScalar<uint5>              ∈ [2, 22]
return %2                                     ^ these are       ^^^^^^^
                                                the assigned    based on
                                                bit-widths      these bounds
```

However, adding integers with different bit-widths (for example, 3-bit and 4-bit numbers) directly isn't possible due to differences in encoding, as shown below:

```
D: data
N: noise

3-bit number
------------
D2 D1 D0 0 0 0 ... 0 0 0 N N N N

4-bit number
------------
D3 D2 D1 D0 0 0 0 ... 0 0 0 N N N N
```

When you add a 3-bit number and a 4-bit number, the result is a 5-bit number with a different encoding: 

```
5-bit number
------------
D4 D3 D2 D1 D0 0 0 0 ... 0 0 0 N N N N
```
## Bit-width assignment with graph processing
To address these encoding differences, a graph processing step called bit-width assignment is performed. This step updates the graph's bit-widths to ensure compatibility with Fully Homomorphic Encryption (FHE). 

After this step, the graph might look like this:


```
%0 = x                  # EncryptedScalar<uint5>
%1 = y                  # EncryptedScalar<uint5>
%2 = add(%0, %1)        # EncryptedScalar<uint5>
return %2
```
## Encoding flexibility with Table Lookup

Most operations cannot change the encoding, requiring the input and output bit-widths to remain the same. However, the table lookup operation can change the encoding. For example, consider the following graph:

```
%0 = x                    # EncryptedScalar<uint2>        ∈ [0, 3]
%1 = y                    # EncryptedScalar<uint5>        ∈ [0, 31]
%2 = 2                    # ClearScalar<uint2>            ∈ [2, 2]
%3 = power(%0, %2)        # EncryptedScalar<uint4>        ∈ [0, 9]
%4 = add(%3, %1)          # EncryptedScalar<uint6>        ∈ [1, 39]
return %4
```

This graph represents the computation `(x**2) + y` where `x` is 2-bits and `y` is 5-bits. Without the ability to change encodings, all bit-widths would need to be adjusted to 6-bits. However, since the encoding can change, bit-widths are assigned more efficiently:

```
%0 = x                    # EncryptedScalar<uint2>        ∈ [0, 3]
%1 = y                    # EncryptedScalar<uint6>        ∈ [0, 31]
%2 = 2                    # ClearScalar<uint2>            ∈ [2, 2]
%3 = power(%0, %2)        # EncryptedScalar<uint6>        ∈ [0, 9]
%4 = add(%3, %1)          # EncryptedScalar<uint6>        ∈ [1, 39]
return %4
```

In this case, `x` remains a 2-bit integer, but the Table Lookup result and `y` are set to 6-bits to allow for the addition.

## Enabling and disabling multi-precision
This approach to bit-width assignment is known as multi-precision and is enabled by default. To disable multi-precision and enforce a single precision across the circuit, use the `single_precision=True` configuration option.

