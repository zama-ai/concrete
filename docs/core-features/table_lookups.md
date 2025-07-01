# Table lookup


This document introduces the concept of Table Lookups (TLUs) in **Concrete**, covers the basic TLU usage, performance considerations, and some basic techniques for optimizing TLUs in encrypted computations. For more advanced TLU usage, refer to the [Table Lookup advanced](table_lookups_advanced.md) section


In TFHE, there exists mainly two operations: the linear operations, such as additions, subtractions, multiplications by integer, and the non-linear operations. Non-linear operations are achieved with Table Lookups (TLUs). 

## Performance 

When using TLUs in Concrete, the most crucial factor for speed is the bit-width of the TLU. The smaller the bit width, the faster the corresponding FHE operation. Therefore, you should reduce the size of inputs to the lookup tables whenever possible. At the end of this document, we discuss methods for truncating or rounding entries to decrease the effective input size, further improving TLU performance.

## Direct TLU

A direct TLU performs operations in the form of `y = T[i]`, where `T` is a table and `i` is an index. You can define the table using `fhe.LookupTable` and apply it to scalars or tensors.

### Scalar lookup
```python
from concrete import fhe

table = fhe.LookupTable([2, -1, 3, 0])

@fhe.compiler({"x": "encrypted"})
def f(x):
    return table[x]

inputset = range(4)
circuit = f.compile(inputset)

assert circuit.encrypt_run_decrypt(0) == table[0] == 2
assert circuit.encrypt_run_decrypt(1) == table[1] == -1
assert circuit.encrypt_run_decrypt(2) == table[2] == 3
assert circuit.encrypt_run_decrypt(3) == table[3] == 0
```
### Tensor lookup
```python
from concrete import fhe
import numpy as np

table = fhe.LookupTable([2, -1, 3, 0])


@fhe.compiler({"x": "encrypted"})
def f(x):
    return table[x]


inputset = [np.random.randint(0, 4, size=(2, 3)) for _ in range(10)]
circuit = f.compile(inputset)

sample = [
    [0, 1, 3],
    [2, 3, 1],
]
expected_output = [
    [2, -1, 0],
    [3, 0, -1],
]
actual_output = circuit.encrypt_run_decrypt(np.array(sample))

assert np.array_equal(actual_output, expected_output)
```

The `LookupTable` behaves like Python's array indexing, where negative indices access elements from the end of the table.

## Multi TLU

A multi TLU is used to apply different elements of the input to different tables (e.g., square the first column, cube the second column):

```python
from concrete import fhe
import numpy as np

squared = fhe.LookupTable([i ** 2 for i in range(4)])
cubed = fhe.LookupTable([i ** 3 for i in range(4)])

table = fhe.LookupTable([
    [squared, cubed],
    [squared, cubed],
    [squared, cubed],
])

@fhe.compiler({"x": "encrypted"})
def f(x):
    return table[x]

inputset = [np.random.randint(0, 4, size=(3, 2)) for _ in range(10)]
circuit = f.compile(inputset)

sample = [
    [0, 1],
    [2, 3],
    [3, 0],
]
expected_output = [
    [0, 1],
    [4, 27],
    [9, 0]
]
actual_output = circuit.encrypt_run_decrypt(np.array(sample))

assert np.array_equal(actual_output, expected_output)
```

### Transparent TLU

In many cases, you won't need to define your own TLUs, as Concrete will set them for you.

```python
from concrete import fhe

@fhe.compiler({"x": "encrypted"})
def f(x):
    return x ** 2

inputset = range(4)
circuit = f.compile(inputset, show_mlir = True)

assert circuit.encrypt_run_decrypt(0) == 0
assert circuit.encrypt_run_decrypt(1) == 1
assert circuit.encrypt_run_decrypt(2) == 4
assert circuit.encrypt_run_decrypt(3) == 9
```

Note that this kind of TLU is compatible with the TLU options, particularly with rounding and truncating which are explained below.

{% hint style="info" %}

[fhe.univariate](../core-features/extensions.md#fheunivariatefunction) and [fhe.multivariate](../core-features/extensions.md#fhemultivariatefunction) extensions are convenient ways to perform more complex operations as transparent TLUs.

{% endhint %}

## Optimizing input size

Reducing the bit size of TLU inputs is essential for execution efficiency, as mentioned in the previous [performance](#performance) section.  One effective method is to replace the table lookup `y = T[i]` by some `y = T'[i']`, where `i'` only has the most significant bits of `i` and `T'` is a much shorter table. This approach can significantly speed up the TLU while maintaining acceptable accuracy in many applications, such as machine learning. 

In this section, we introduce two basic techniques: [truncating](#truncating) or [rounding](#rounding). You can find more in-depth explanation and other advanced techniques of optimization in the [TLU advanced documentation](table_lookups_advanced.md).

### Truncating
The first option is to set `i'` as the truncation of `i`. In this method, we just take the most significant bits of `i`. This is done with `fhe.truncate_bit_pattern`.

```python
from concrete import fhe
import numpy as np

table = fhe.LookupTable([i**2 for i in range(16)])
lsbs_to_remove = 1


@fhe.compiler({"x": "encrypted"})
def f(x):
    return table[fhe.truncate_bit_pattern(x, lsbs_to_remove)]


inputset = range(16)
circuit = f.compile(inputset)

for i in range(16):
    rounded_i = int(i / 2**lsbs_to_remove) * 2**lsbs_to_remove

    assert circuit.encrypt_run_decrypt(i) == rounded_i**2
```
### Rounding

The second option is to set `i` as the rounded value of `i`. In this method, we take the most significant bits of `i` and round up by 1 if the most significant ignored bit is 1.  This is done with `fhe.round_bit_pattern`. 

However, this approach can be slightly more complex, as rounding might result in an index that exceeds the original table's bounds. To handle this, we expand the original table by one additional index:

```python
from concrete import fhe
import numpy as np

table = fhe.LookupTable([i**2 for i in range(17)])
lsbs_to_remove = 1


def our_round(x):
    float_part = x - np.floor(x)
    if float_part < 0.5:
        return int(np.floor(x))
    return int(np.ceil(x))


@fhe.compiler({"x": "encrypted"})
def f(x):
    return table[fhe.round_bit_pattern(x, lsbs_to_remove)]


inputset = range(16)
circuit = f.compile(inputset)

for i in range(16):
    rounded_i = our_round(i * 1.0 / 2**lsbs_to_remove) * 2**lsbs_to_remove

    assert (
        circuit.encrypt_run_decrypt(i) == rounded_i**2
    ), f"Miscomputation {i=} {circuit.encrypt_run_decrypt(i)} {rounded_i**2}"
```
### Approximate rounding
For further optimizations, the `fhe.round_bit_pattern` function has an `exactness=fhe.Exactness.APPROXIMATE` option, which allows for faster computations at the cost of minor differences between cleartext and encrypted results:

```python
from concrete import fhe
import numpy as np

table = fhe.LookupTable([i**2 for i in range(17)])
lsbs_to_remove = 1


@fhe.compiler({"x": "encrypted"})
def f(x):
    return table[fhe.round_bit_pattern(x, lsbs_to_remove, exactness=fhe.Exactness.APPROXIMATE)]


inputset = range(16)
circuit = f.compile(inputset)

for i in range(16):
    lower_i = np.floor(i * 1.0 / 2**lsbs_to_remove) * 2**lsbs_to_remove
    upper_i = np.ceil(i * 1.0 / 2**lsbs_to_remove) * 2**lsbs_to_remove

    assert circuit.encrypt_run_decrypt(i) in [
        lower_i**2,
        upper_i**2,
    ], f"Miscomputation {i=} {circuit.encrypt_run_decrypt(i)} {[lower_i**2, upper_i**2]}"
```
