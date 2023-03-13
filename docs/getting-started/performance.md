# Performance

One of the most common operations in **Concrete** is `Table Lookups` (TLUs). All operations except addition, subtraction, multiplication with non-encrypted values, tensor manipulation operations, and a few operations built with those primitive operations (e.g. matmul, conv) are converted to table lookups under the hood:

```python
from concrete import fhe

@fhe.compiler({"x": "encrypted"})
def f(x):
    return x ** 2

inputset = range(2 ** 4)
circuit = f.compile(inputset)
```

is exactly the same as

```python
from concrete import fhe

table = fhe.LookupTable([x ** 2 for x in range(2 ** 4)])

@fhe.compiler({"x": "encrypted"})
def f(x):
    return table[x]

inputset = range(2 ** 4)
circuit = f.compile(inputset)
```

Table lookups are very flexible! They allow Concrete to support many operations, but they are expensive. The exact cost depends on many variables (hardware used, error probability, etc.) but they are always much more expensive compared to other operations. Therefore, you should try to avoid them as much as possible. In most cases, it's not possible to avoid them completely, but you might remove the number of TLUs or replace some of them with other primitive operations.

{% hint style="info" %}
Concrete automatically parallelize TLUs if they are applied to tensors.
{% endhint %}
