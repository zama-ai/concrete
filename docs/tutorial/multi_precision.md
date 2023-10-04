# Multi Precision

Each integer in the circuit has a certain bit-width, which is determined by the inputset. These bit-widths can be observed when graphs are printed:

```
%0 = x                  # EncryptedScalar<uint3>              ∈ [0, 7]
%1 = y                  # EncryptedScalar<uint4>              ∈ [0, 15]
%2 = add(%0, %1)        # EncryptedScalar<uint5>              ∈ [2, 22]
return %2                                     ^ these are       ^^^^^^^
                                                the assigned    based on
                                                bit-widths      these bounds
```

However, it's not possible to add 3-bit and 4-bit numbers together because their encoding is different:

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

The result of such an addition is a 5-bit number, which also has a different encoding:

```
5-bit number
------------
D4 D3 D2 D1 D0 0 0 0 ... 0 0 0 N N N N
```

Because of these encoding differences, we perform a graph processing step called bit-width assignment, which takes the graph and updates the bit-widths to be compatible with FHE.

After this graph processing step, the graph would look like:

```
%0 = x                  # EncryptedScalar<uint5>
%1 = y                  # EncryptedScalar<uint5>
%2 = add(%0, %1)        # EncryptedScalar<uint5>
return %2
```

Most operations cannot change the encoding, which means that the input and output bit-widths need to be the same. However, there is an operation which can change the encoding: the table lookup operation.

Let's say you have this graph:
```
%0 = x                    # EncryptedScalar<uint2>        ∈ [0, 3]
%1 = y                    # EncryptedScalar<uint5>        ∈ [0, 31]
%2 = 2                    # ClearScalar<uint2>            ∈ [2, 2]
%3 = power(%0, %2)        # EncryptedScalar<uint4>        ∈ [0, 9]
%4 = add(%3, %1)          # EncryptedScalar<uint6>        ∈ [1, 39]
return %4
```

This is the graph for `(x**2) + y` where `x` is 2-bits and `y` is 5-bits. If the table lookup operation wasn't able to change the encoding, we'd need to make everything 6-bits. However, since the encoding can be changed, the bit-widths can be assigned like so:

```
%0 = x                    # EncryptedScalar<uint2>        ∈ [0, 3]
%1 = y                    # EncryptedScalar<uint6>        ∈ [0, 31]
%2 = 2                    # ClearScalar<uint2>            ∈ [2, 2]
%3 = power(%0, %2)        # EncryptedScalar<uint6>        ∈ [0, 9]
%4 = add(%3, %1)          # EncryptedScalar<uint6>        ∈ [1, 39]
return %4
```

In this case, we kept `x` as 2-bits, but set the table lookup result and `y` to be 6-bits, so that the addition can be performed.

This style of bit-width assignment is called multi-precision, and it is enabled by default. To disable it and use a single precision across the circuit, you can use the `single_precision=True` configuration option.
