# Table lookup

In TFHE, there exists mainly two operations: the linear operations (additions, subtractions, multiplications by integers) and the rest. And the rest is done with table lookups (TLUs), which means that a lot of things are done with TLU. In this document, we explain briefly, from a user point of view, how TLU can be used. In [the poweruser documentation](table_lookups_advanced.md), we enter a bit more into the details.

## Performance

Before entering into the details on how we can use TLU in Concrete, let us mention the most important parameter for speed here: the smallest the bitwidth of a TLU is, the faster the corresponding FHE operation will be. Which means, in general, the user should try to reduce the size of the inputs to the tables. Also, we propose in the end of this document ways to truncate or round entries, which makes effective inputs smaller and so makes corresponding TLU faster.

## Direct TLU

Direct TLU stands for instructions of the form `y = T[i]`, for some table `T` and some index `i`. One can use the `fhe.LookupTable` to define the table values, and then use it on scalars or tensors.

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

`LookupTable` mimics array indexing in Python, which means if the lookup variable is negative, the table is looked up from the back.

## Multi TLU

Multi TLU stands for instructions of the form `y = T[j][i]`, for some set of tables `T`, some table-index `j` and some index `i`. One can use the `fhe.LookupTable` to define the table values, and then use it on scalars or tensors.

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

For lot of programs, users will not even need to define their table lookup, it will be done
automatically by Concrete.

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

Remark that this kind of TLU is compatible with the TLU options, and in particular, with rounding and
truncating which are explained below.

{% hint style="info" %}

[fhe.univariate](../core-features/extensions.md#fheunivariatefunction) and [fhe.multivariate](../core-features/extensions.md#fhemultivariatefunction) extensions are convenient ways to perform more complex operations as transparent TLUs.

{% endhint %}

### TLU on the most significant bits

As we said in the beginning of this document, bitsize of the inputs of TLU are critical for the efficiency of the execution: the slower they are, the faster the TLU will be. Thus, we have proposed a few mechanisms to reduce this bitsize: the main one is _rounding_ or _truncating_.

For lot of use-cases, like for example in Machine Learning, it is possible to replace the table lookup `y = T[i]` by some `y = T'[i']`, where `i'` only has the most significant bits of `i` and `T'` is a much shorter table, and still maintain a good accuracy of the function. The interest of such a method stands in the fact that, since the table `T'` is much smaller, the corresponding TLU will be done much more quickly.

There are different flavors of doing this in Concrete. We describe them quickly here, and refer the user to the [poweruser documentation](table_lookups_advanced.md) for more explanations.

The first possibility is to set `i'` as the truncation of `i`: here, we just take the most significant bits of `i`. This is done with `fhe.truncate_bit_pattern`.

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

The second possibility is to set `i'` as the rounding of `i`: here, we take the most significant bits of `i`, and increment by 1 if ever the most significant ignored bit is a 1, to round.  This is done with `fhe.round_bit_pattern`. It's however a bit more complicated, since rounding may make us go "out" of the original table. Remark how we enlarged the original table by 1 index.

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

Finally, for `fhe.round_bit_pattern`, there exist an `exactness=fhe.Exactness.APPROXIMATE` option, which can make computations even faster, at the price of having a few (minor) differences between cleartext computations and encrypted computations.

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

