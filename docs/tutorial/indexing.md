# Indexing

## Constant Indexing

Constant indexing refers to the index being static (i.e., known during compilation).

Here are some examples of constant indexing:

### Extracting a single element

```python
import concrete.numpy as cnp
import numpy as np

@cnp.compiler({"x": "encrypted"})
def f(x):
    return x[1]

inputset = [np.random.randint(0, 2 ** 3, size=(3,), dtype=np.uint8) for _ in range(10)]
circuit = f.compile(inputset)

test_input = np.array([4, 2, 6], dtype=np.uint8)
expected_output = 2

assert np.array_equal(circuit.encrypt_run_decrypt(test_input), expected_output)
```

You can use negative indexing.

```python
import concrete.numpy as cnp
import numpy as np

@cnp.compiler({"x": "encrypted"})
def f(x):
    return x[-1]

inputset = [np.random.randint(0, 2 ** 3, size=(3,), dtype=np.uint8) for _ in range(10)]
circuit = f.compile(inputset)

test_input = np.array([4, 2, 6], dtype=np.uint8)
expected_output = 6

assert np.array_equal(circuit.encrypt_run_decrypt(test_input), expected_output)
```

You can use multidimensional indexing as well.

```python
import concrete.numpy as cnp
import numpy as np

@cnp.compiler({"x": "encrypted"})
def f(x):
    return x[-1, 1]

inputset = [np.random.randint(0, 2 ** 3, size=(3, 2), dtype=np.uint8) for _ in range(10)]
circuit = f.compile(inputset)

test_input = np.array([[4, 2], [1, 5], [7, 6]], dtype=np.uint8)
expected_output = 6

assert np.array_equal(circuit.encrypt_run_decrypt(test_input), expected_output)
```

### Extracting a slice

```python
import concrete.numpy as cnp
import numpy as np

@cnp.compiler({"x": "encrypted"})
def f(x):
    return x[1:4]

inputset = [np.random.randint(0, 2 ** 3, size=(5,), dtype=np.uint8) for _ in range(10)]
circuit = f.compile(inputset)

test_input = np.array([4, 2, 6, 1, 7], dtype=np.uint8)
expected_output = np.array([2, 6, 1], dtype=np.uint8)

assert np.array_equal(circuit.encrypt_run_decrypt(test_input), expected_output)
```

You can use multidimensional slicing as well.

{% hint style='tip' %}
There are certain limitations of slicing due to MLIR. So if you stumple into `RuntimeError: Compilation failed: Failed to lower to LLVM dialect`, know that we are aware of it, and we are trying to make such cases compilable.
{% endhint %}

## Dynamic Indexing

Dynamic indexing refers to the index being dynamic (i.e., can change during runtime).
Such indexing is especially useful for things like decision trees.
Unfortunately, we don't support dynamic indexing for the time being.
