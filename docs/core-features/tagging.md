# Tagging

This document explains the concept of tagging, which is a debugging tool to make a link between the user's Python code and the Concrete MLIR circuits. Such a link can be useful when an issue is raised by the compiler on some MLIR, to know which Python code it corresponds to.

When you have big circuits, keeping track of which node corresponds to which part of your code becomes difficult. A tagging system can simplify such situations:

```python
def g(z):
    with fhe.tag("def"):
        a = 120 - z
        b = a // 4
    return b


def f(x):
    with fhe.tag("abc"):
        x = x * 2
        with fhe.tag("foo"):
            y = x + 42
        z = np.sqrt(y).astype(np.int64)

    return g(z + 3) * 2
```

When you compile `f` with inputset of `range(10)`, you get the following graph:

```
 %0 = x                            # EncryptedScalar<uint4>        ∈ [0, 9]
 %1 = 2                            # ClearScalar<uint2>            ∈ [2, 2]            @ abc
 %2 = multiply(%0, %1)             # EncryptedScalar<uint5>        ∈ [0, 18]           @ abc
 %3 = 42                           # ClearScalar<uint6>            ∈ [42, 42]          @ abc.foo
 %4 = add(%2, %3)                  # EncryptedScalar<uint6>        ∈ [42, 60]          @ abc.foo
 %5 = subgraph(%4)                 # EncryptedScalar<uint3>        ∈ [6, 7]            @ abc
 %6 = 3                            # ClearScalar<uint2>            ∈ [3, 3]
 %7 = add(%5, %6)                  # EncryptedScalar<uint4>        ∈ [9, 10]
 %8 = 120                          # ClearScalar<uint7>            ∈ [120, 120]        @ def
 %9 = subtract(%8, %7)             # EncryptedScalar<uint7>        ∈ [110, 111]        @ def
%10 = 4                            # ClearScalar<uint3>            ∈ [4, 4]            @ def
%11 = floor_divide(%9, %10)        # EncryptedScalar<uint5>        ∈ [27, 27]          @ def
%12 = 2                            # ClearScalar<uint2>            ∈ [2, 2]
%13 = multiply(%11, %12)           # EncryptedScalar<uint6>        ∈ [54, 54]
return %13

Subgraphs:

    %5 = subgraph(%4):

        %0 = input                         # EncryptedScalar<uint2>          @ abc.foo
        %1 = sqrt(%0)                      # EncryptedScalar<float64>        @ abc
        %2 = astype(%1, dtype=int_)        # EncryptedScalar<uint1>          @ abc
        return %2
```

If you get an error, you'll see exactly where the error occurred (e.g., which layer of the neural network, if you tag layers).

{% hint style="info" %}
In the future, we plan to use tags for additional features (e.g., to measure performance of tagged regions), so it's a good idea to start utilizing them for big circuits.
{% endhint %}
