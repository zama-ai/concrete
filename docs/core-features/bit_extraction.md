# Bit Extraction
This document provides an overview of the bit extraction feature in **Concrete**, including usage examples, limitations, and performance considerations. 

## Overview

Bit extraction could be useful in some applications that require directly manipulating bits of integers. Bit extraction allows you to extract a specific slice of bits from an integer, where index 0 corresponds to the least significant bit (LSB). The cost of this operation increases with the index of the highest significant bit you wish to extract.

{% hint style="warning" %}
Bit extraction only works in the `Native` encoding, which is usually selected when all table lookups in the circuit are less than or equal to 8 bits.
{% endhint %}

## Extracting a specific bit

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

## Extracting multiple bits with slices
You can use slices for indexing `fhe.bits(value)` :

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

Bit extraction supports slices with negative steps:

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
## Bit extraction with signed integers
Bit extraction supports signed integers:

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

### Use case example
Here's a practical example that uses bit extraction to determine if a number is even:

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

It prints:

```
13 is odd
0 is even
-15 is odd
2 is even
-6 is even
```

## Limitations
- **Negative indexing is not supported:** Bits extraction using negative indices is not supported, such as `fhe.bits(x)[-1]`.
    - This is because the bit-width of `x` is unknown before inputset evaluation, making it impossible to determine the correct bit to extract.
- **Reverse slicing requires explicit starting bit:** When extracting bits in reverse order (using a negative step), the start bit must be specified, for example, `fhe.bits(x)[::-1]` is not supported.
- **Signed integer slicing requires explicit stopping bit**: For signed integers, when using slices, the stop bit must be explicitly provided, for example, `fhe.bits(x)[1:]` is not supported.
- **Float bit extraction is not supported**: While Concrete supports floats to some extent, bit extraction is not possible on float types.

## Performance considerations

### A Chain of individual bit extractions

Extracting a specific bit requires clearing all the preceding lower bits. This involves extracting these previous bits as intermediate values and then subtracting them from the input.

**Implications:**

* Bits are extracted sequentially, starting from the least significant bit to the more significant ones. The cost is proportional to the index of the highest extracted bit plus one.
* No parallelization is possible. The computation time is proportional to the cost, independent of the number of CPUs.

**Examples:**

* Extracting `fhe.bits(x)[4]` is approximately five times costlier than extracting `fhe.bits(x)[0]`.
* Extracting `fhe.bits(x)[4]` takes around five times more wall clock time than `fhe.bits(x)[0]`.
* The cost of extracting `fhe.bits(x)[0:5]` is almost the same as that of `fhe.bits(x)[5]`.

### Reuse of Intermediate Extracted Bits

Common sub-expression elimination is applied to intermediate extracted bits.

**Implications:**

* The overall cost for a series of `fhe.bits(x)[m:n]` calls on the same input `x` is almost equivalent to the cost of the single most computationally expensive extraction in the series, i.e. `fhe.bits(x)[n]`.
* The order of extraction in that series does not affect the overall cost.

**Example**:

The combined operation `fhe.bit(x)[3] + fhe.bit(x)[2] + fhe.bit(x)[1]` has almost the same cost as `fhe.bits(x)[3]`.

### TLUs of 1b input precision

Each extracted bit incurs a cost of approximately one TLU of 1-bit input precision. Therefore, `fhe.bits(x)[0]` is generally faster than any other TLU operation.
