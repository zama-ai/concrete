# Table Lookups

In this tutorial, we will review how to perform direct table lookups in **Concrete**.

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
