# Table lookups (advanced)

This document details the management of Table Lookups(TLU) within Concrete for advanced usage. For a simpler guide, refer to the [Table Lookup Basics](table_lookups.md).

One of the most common operations in **Concrete** are `Table Lookups` (TLUs). All operations except addition, subtraction, multiplication with non-encrypted values, tensor manipulation operations, and a few operations built with those primitive operations (e.g. matmul, conv) are converted to Table Lookups under the hood.

Table Lookups are very flexible. They allow Concrete to support many operations, but they are expensive. The exact cost depends on many variables (hardware used, error probability, etc.), but they are always much more expensive compared to other operations. You should try to avoid them as much as possible. It's not always possible to avoid them completely, but you might remove the number of TLUs or replace some of them with other primitive operations.

{% hint style="info" %}
Concrete automatically parallelizes TLUs if they are applied to tensors.
{% endhint %}

## Direct table lookup

**Concrete** provides a `LookupTable` class to create your own tables and apply them in your circuits.

{% hint style="info" %}
`LookupTable`s can have any number of elements. Let's call the number of elements **N**. As long as the lookup variable is within the range \[-**N**, **N**), the Table Lookup is valid.

If you go outside of this range, you will receive the following error:

```
IndexError: index 10 is out of bounds for axis 0 with size 6
```
{% endhint %}

### With scalars.

You can create the lookup table using a list of integers and apply it using indexing:

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

### With tensors.

When you apply a table lookup to a tensor, the scalar table lookup is applied to each element of the tensor:

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

for i in range(2):
    for j in range(3):
        assert actual_output[i][j] == expected_output[i][j] == table[sample[i][j]]
```

### With negative values.

`LookupTable` mimics array indexing in Python, which means if the lookup variable is negative, the table is looked up from the back:

```python
from concrete import fhe

table = fhe.LookupTable([2, -1, 3, 0])

@fhe.compiler({"x": "encrypted"})
def f(x):
    return table[-x]

inputset = range(1, 5)
circuit = f.compile(inputset)

assert circuit.encrypt_run_decrypt(1) == table[-1] == 0
assert circuit.encrypt_run_decrypt(2) == table[-2] == 3
assert circuit.encrypt_run_decrypt(3) == table[-3] == -1
assert circuit.encrypt_run_decrypt(4) == table[-4] == 2
```

## Direct multi-table lookup

If you want to apply a different lookup table to each element of a tensor, you can have a `LookupTable` of `LookupTable`s:

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

for i in range(3):
    for j in range(2):
        if j == 0:
            assert actual_output[i][j] == expected_output[i][j] == squared[sample[i][j]]
        else:
            assert actual_output[i][j] == expected_output[i][j] == cubed[sample[i][j]]
```

In this example, we applied a `squared` table to the first column and a `cubed` table to the second column.

## Fused table lookup

**Concrete** tries to fuse some operations into table lookups automatically so that lookup tables don't need to be created manually:

```python
from concrete import fhe
import numpy as np

@fhe.compiler({"x": "encrypted"})
def f(x):
    return (42 * np.sin(x)).astype(np.int64) // 10

inputset = range(8)
circuit = f.compile(inputset)

for x in range(8):
    assert circuit.encrypt_run_decrypt(x) == f(x)
```

{% hint style="info" %}
All lookup tables need to be from integers to integers. So, without `.astype(np.int64)`, **Concrete** will not be able to fuse.
{% endhint %}

The function is first traced into:

![](../\_static/tutorials/table-lookup/1.initial.graph.png)

**Concrete** then fuses appropriate nodes:

![](../\_static/tutorials/table-lookup/3.final.graph.png)

{% hint style="info" %}
Fusing makes the code more readable and easier to modify, so try to utilize it over manual `LookupTable`s as much as possible.
{% endhint %}

## Using automatically created table lookup

We refer the users to [this page](extensions.md) for explanations about `fhe.univariate(function)` and `fhe.multivariate(function)` features, which are convenient ways to use automatically created table lookup.

## Table lookup exactness

TLUs are performed with an FHE operation called `Programmable Bootstrapping` (PBS). PBSs have a certain probability of error: when these errors happen, it results in inaccurate results.

Let's say you have the table:

```python
lut = [0, 1, 4, 9, 16, 25, 36, 49, 64]
```

And you perform a Table Lookup using `4`. The result you should get is `lut[4] = 16`, but because of the possibility of error, you could get any other value in the table.

The probability of this error can be configured through the `p_error` and `global_p_error` configuration options. The difference between these two options is that, `p_error` is for individual TLUs but `global_p_error` is for the whole circuit.

If you set `p_error` to `0.01`, for example, it means every TLU in the circuit will have a 99% chance (or more) of being exact. If there is a single TLU in the circuit, it corresponds to `global_p_error = 0.01` as well. But if we have 2 TLUs, then `global_p_error` would be higher: that's `1 - (0.99 * 0.99) ~= 0.02 = 2%`.

If you set `global_p_error` to `0.01`, the whole circuit will have at most 1% probability of error, no matter how many Table Lookups are included (which means that `p_error` will be smaller than `0.01` if there are more than a single TLU).

If you set both of them, both will be satisfied. Essentially, the stricter one will be used.

By default, both `p_error` and `global_p_error` are set to `None`, which results in a `global_p_error` of `1 / 100_000` being used.

Feel free to play with these configuration options to pick the one best suited for your needs! See [How to Configure](../guides/configure.md) to learn how you can set a custom `p_error` and/or `global_p_error`.

{% hint style="info" %}
Configuring either of those variables impacts compilation and execution times (compilation, keys generation, circuit execution) and space requirements (size of the keys on disk and in memory). Lower error probabilities result in longer compilation and execution times and larger space requirements.
{% endhint %}

## Table lookup performance

PBSs are very expensive, in terms of computations. Fortunately, it is sometimes possible to replace PBS by [rounded PBS](rounding.md), [truncate PBS](truncating.md) or even [approximate PBS](rounding.md). These TLUs have a slightly different semantic, but are very useful in cases like machine learning for more efficiency without drop of accuracy.
