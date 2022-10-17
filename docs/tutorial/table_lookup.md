# Table Lookups

In this tutorial, we will review the ways to perform direct table lookups in **Concrete-Numpy**.

## Direct table lookup

**Concrete-Numpy** provides a `LookupTable` class for you to create your own tables and apply them in your circuits.

{% hint style="info" %}
`LookupTable`s can have any number of elements. Let's call them **N**. As long as the lookup variable is in range \[-**N**, **N**), table lookup is valid.

If you go out of bounds of this range, you will get the following error:

```
IndexError: index 10 is out of bounds for axis 0 with size 6
```
{% endhint %}

{% hint style="info" %}
The number of elements in the lookup table doesn't affect performance in any way.
{% endhint %}

### With scalars.

You can create the lookup table using a list of integers and apply it using indexing:

```python
import concrete.numpy as cnp

table = cnp.LookupTable([2, -1, 3, 0])

@cnp.compiler({"x": "encrypted"})
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

When you apply the table lookup to a tensor, you apply the scalar table lookup to each element of the tensor:

```python
import concrete.numpy as cnp
import numpy as np

table = cnp.LookupTable([2, -1, 3, 0])

@cnp.compiler({"x": "encrypted"})
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
import concrete.numpy as cnp

table = cnp.LookupTable([2, -1, 3, 0])

@cnp.compiler({"x": "encrypted"})
def f(x):
    return table[-x]

inputset = range(1, 5)
circuit = f.compile(inputset)

assert circuit.encrypt_run_decrypt(1) == table[-1] == 0
assert circuit.encrypt_run_decrypt(2) == table[-2] == 3
assert circuit.encrypt_run_decrypt(3) == table[-3] == -1
assert circuit.encrypt_run_decrypt(4) == table[-4] == 2
```

## Direct multi table lookup

In case you want to apply a different lookup table to each element of a tensor, you can have a `LookupTable` of `LookupTable`s:

```python
import concrete.numpy as cnp
import numpy as np

squared = cnp.LookupTable([i ** 2 for i in range(4)])
cubed = cnp.LookupTable([i ** 3 for i in range(4)])

table = cnp.LookupTable([
    [squared, cubed],
    [squared, cubed],
    [squared, cubed],
])

@cnp.compiler({"x": "encrypted"})
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

In this example, we applied a `squared` table to the first column and a `cubed` table to the second one.

## Fused table lookup

**Concrete-Numpy** tries to fuse some operations into table lookups automatically, so you don't need to create the lookup tables manually:

```python
import concrete.numpy as cnp
import numpy as np

@cnp.compiler({"x": "encrypted"})
def f(x):
    return (42 * np.sin(x)).astype(np.int64) // 10

inputset = range(8)
circuit = f.compile(inputset)

for x in range(8):
    assert circuit.encrypt_run_decrypt(x) == f(x)
```

{% hint style="info" %}
All lookup tables need to be from integers to integers. So, without `.astype(np.int64)`, **Concrete-Numpy** will not be able to fuse.
{% endhint %}

The function is first traced into:

![](../\_static/tutorials/table-lookup/1.initial.graph.png)

Then, **Concrete-Numpy** fuses appropriate nodes:

![](../\_static/tutorials/table-lookup/3.final.graph.png)

{% hint style="info" %}
Fusing makes the code more readable and easier to modify, so try to utilize it over manual `LookupTable`s as much as possible.
{% endhint %}
