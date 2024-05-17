# Bit extraction

This document explains bit extraction, a feature of the **Concrete** library for manipulating individual bits within encrypted integers.

## Introduction

Bit extraction allows you to extract a slice of bits from an integer. This operation is useful for applications that require direct manipulations of bits of integers.

**Concrete**'s `fhe.bits(x)` function takes an encrypted integer `x` as input and returns a list representing the individual bits of `x`. Index 0 corresponds to the least significant bit of `x`. The cost of this operation increases as the index goes higher.

{% hint style="warning" %}
Bit extraction only works in the `Native` encoding, which is typically used when all Table Lookups in the circuit are less than or equal to 8 bits.
{% endhint %}

## 1. Extracting individual bits

You can access specific bits by indexing the list returned by `fhe.bits(value)`:

```python
from concrete import fhe

@fhe.compiler({"x": "encrypted"})
def f(x):
    return fhe.bits(x)[0], fhe.bits(x)[3]

inputset = range(32)
circuit = f.compile(inputset)

assert circuit.encrypt_run_decrypt(0b_00000) == (0, 0)
assert circuit.encrypt_run_decrypt(0b_00001) == (1, 0)

assert circuit.encrypt_run_decrypt(0b_01100) == (0, 1)
assert circuit.encrypt_run_decrypt(0b_01101) == (1, 1)
```

## 2. Extracting slices of bits

You can also efficiently extract a slice of bits by indexing `fhe.bits(value)`:

```python
from concrete import fhe

@fhe.compiler({"x": "encrypted"})
def f(x):
    return fhe.bits(x)[1:4]

inputset = range(32)
circuit = f.compile(inputset)

assert circuit.encrypt_run_decrypt(0b_01101) == 0b_110
assert circuit.encrypt_run_decrypt(0b_01011) == 0b_101
```

This feature supports extracting slices with negative steps as well:

```python
from concrete import fhe

@fhe.compiler({"x": "encrypted"})
def f(x):
    return fhe.bits(x)[3:0:-1]

inputset = range(32)
circuit = f.compile(inputset)

assert circuit.encrypt_run_decrypt(0b_01101) == 0b_011
assert circuit.encrypt_run_decrypt(0b_01011) == 0b_101
```

## 3. Extracting bits in signed integers

Here's how to extract bits in signed integers:

```python
from concrete import fhe

@fhe.compiler({"x": "encrypted"})
def f(x):
    return fhe.bits(x)[1:3]

inputset = range(-16, 16)
circuit = f.compile(inputset)

assert circuit.encrypt_run_decrypt(-14) == 0b_01  # -14 == 0b_10010 (in two's complement)
assert circuit.encrypt_run_decrypt(-12) == 0b_10  # -12 == 0b_10100 (in two's complement)
```

## Example use case

This example shows how to use bit extraction to check if an encrypted integer is even:

```python
import numpy as np
from concrete import fhe

@fhe.compiler({"x": "encrypted"})
def is_even(x):
    return 1 - fhe.bits(x)[0]

inputset = [
    np.random.randint(-16, 16, size=(5,))
    for _ in range(100)
]
circuit = is_even.compile(inputset)

sample = np.random.randint(-16, 16, size=(5,))
for value, value_is_even in zip(sample, circuit.encrypt_run_decrypt(sample)):
    print(f"{value} is {'even' if value_is_even else 'odd'}")
```

This function prints:

```
13 is odd
0 is even
-15 is odd
2 is even
-6 is even
```

## Limitations

### Unknown bit width

When extracting bits with `fhe.bits(x)`, **Concrete** can't determine the bit width of `x` before evaluating the input set, thus it doesn't support the indexing in the following ways:

* **Negative indexing**: expressions with negative indexing such as `fhe.bits(x)[-1]` or `fhe.bits(x)[-4:-1]` are not supported.

{% hint style="info" %}
Explanations:

Let's take an example of `x == 10 == 0b_000...0001010` to perform `fhe.bits(x)[-1]`. If the value is a 4-bit `0b_1010`, the result would be `1`, but if the value is a 6-bit `0b_001010`, the result would be `0`. Since **Concrete** can't know the bit width of `x` before inputset evaluation, it cannot calculate `fhe.bits(x)[-1].`
{% endhint %}

* **Slices with negative steps**: when extracting bits using slices in reverse order (step < 0), you need to specify the starting bit explicitly. For example, slices like `fhe.bits(x)[::-1]` or `fhe.bits(x)[:2:-1]` are not supported.
* **Slices for signed integers**: when extracting bits of signed values using slices, you need to specify the stopping bit explicitly. For example, slices like `fhe.bits(x)[1:]` or `fhe.bits(x)[1::2]` are not supported for example.

{% hint style="info" %}
Explanations:

Signed integers use [two's complement](https://en.wikipedia.org/wiki/Two's\_complement) representation. In this representation, negative values have their most significant bits set to 1, for example, `-1 == 0b_11111`, `-2 == 0b_11110`, `-3 == 0b_11101`. The problem is that extracting bits always returns a positive value, for example, `fhe.bits(-1)[1:3] == 0b_11 == 3`. This means if you were to do `fhe.bits(x)[1:]` where `x == -1`, if `x` is 4 bits, the result would be `0b_111 == 7`, but if `x` is 5 bits the result would be `0b_1111 == 15`. Since **Concrete** can't know the bit width of `x` before inputset evaluation, it cannot calculate `fhe.bits(x)[1:]`.
{% endhint %}

### Unsupported data types

* **Floats**: While Concrete partially supports floats as a data type, extracting individual bits from floats is not currently supported.

## Performance considerations

### How bit extraction works

**Concrete** clears all the preceding lower bits when extracting a specific bit from an integer. This involves extracting these previous bits as intermediate values and then subtracting them from the input.

Bit extraction happens sequentially, starting from the least significant bit to the more significant ones. Therefore, the more significant the bit you index, the more the computation costs.

For example:

* Extracting `fhe.bits(x)[4]` costs approximately 5 times more than extracting `fhe.bits(x)[0]`. Therefore, extracting `fhe.bits(x)[4]` takes around 5 times as long as `fhe.bits(x)[0]` in clock time.
* The cost of extracting `fhe.bits(x)[0:5]` is almost the same as `fhe.bits(x)[5]`.

{% hint style="info" %}
Parallelization is not possible during bit extraction. The computation time is directly proportional to the cost, regardless of the number of CPUs.
{% endhint %}

### Optimization

**Concrete** reuses the intermediate extracted bits to optimize the performance. It means that the overall cost of a series of calls to the same input `x` - `fhe.bits(x)[m:n]`  is almost equivalent to the cost of the single most computationally expensive extraction in the series - `fhe.bits(x)[n]`. The order of extraction in that series does affect the overall cost, **Concrete** identifies the most expensive extraction and reuses the bits from there.

For example, the combined operation `fhe.bit(x)[3] + fhe.bit(x)[2] + fhe.bit(x)[1]` has almost the same cost as `fhe.bits(x)[3]`.

### TLUs of 1-bit input precision

Each extracted bit incurs a cost of approximately one TLU of 1-bit input precision. Therefore, `fhe.bits(x)[0]` is generally faster than any other TLU operation.
