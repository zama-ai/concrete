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

Furthermore, the result is a 5-bit number, so it has a different encoding as well:

```
5-bit number
------------
D4 D3 D2 D1 D0 0 0 0 ... 0 0 0 N N N N
```

Because of this encoding difference, we do a graph processing step called bit-width assignment, which takes the graph and updates bit-widths in the graph to be compatible with FHE.

After this graph processing pass, the graph would look like:

```
%0 = x                  # EncryptedScalar<uint5>
%1 = y                  # EncryptedScalar<uint5>
%2 = add(%0, %1)        # EncryptedScalar<uint5>
return %2
```

Most operations cannot change the encoding, so they need to share bit-width between their inputs and their outputs but there is a very important operation which can change the encoding, and it's the table lookup operation.

Let's say you have this graph:
```
%0 = x                    # EncryptedScalar<uint2>        ∈ [0, 3]
%1 = y                    # EncryptedScalar<uint5>        ∈ [0, 31]
%2 = 2                    # ClearScalar<uint2>            ∈ [2, 2]
%3 = power(%0, %2)        # EncryptedScalar<uint4>        ∈ [0, 9]
%4 = add(%3, %1)          # EncryptedScalar<uint6>        ∈ [1, 39]
return %4
```

This is the graph for `(x**2) + y` where `x` is 2-bits and `y` is `5-bits. If the table lookup operation wasn't able to change the encoding, we'd need to make everything 6-bits but because they can, bit-widths can be assigned like so:

```
%0 = x                    # EncryptedScalar<uint2>        ∈ [0, 3]
%1 = y                    # EncryptedScalar<uint6>        ∈ [0, 31]
%2 = 2                    # ClearScalar<uint2>            ∈ [2, 2]
%3 = power(%0, %2)        # EncryptedScalar<uint6>        ∈ [0, 9]
%4 = add(%3, %1)          # EncryptedScalar<uint6>        ∈ [1, 39]
return %4
```

In this case, we kept `x` 2-bit, but set the table lookup result and `y` to 6-bits, so the addition can be performed.

This style of bit-width assignment is called multi-precision. Unfortunately, it's disabled by default at the moment.

To enable it, you can use `single_precision=False` configuration option.

{% hint style="info" %}
Multi precision will become the default at one point in the near future!
{% endhint %}
