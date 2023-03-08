# Extensions

**Concrete-Numpy** tries to support **NumPy** as much as possible, but due to some technical limitations, not everything can be supported. On top of that, there are some things **NumPy** lack, which are useful. In some of these situations, we provide extensions in **Concrete-Numpy** to improve your experience.

## cnp.zero()

Allows you to create encrypted scalar zero:

```python
import concrete.numpy as cnp
import numpy as np

@cnp.compiler({"x": "encrypted"})
def f(x):
    z = cnp.zero()
    return x + z

inputset = range(10)
circuit = f.compile(inputset)

for x in range(10):
    assert circuit.encrypt_run_decrypt(x) == x
```

## cnp.zeros(shape)

Allows you to create encrypted tensor of zeros:

```python
import concrete.numpy as cnp
import numpy as np

@cnp.compiler({"x": "encrypted"})
def f(x):
    z = cnp.zeros((2, 3))
    return x + z

inputset = range(10)
circuit = f.compile(inputset)

for x in range(10):
    assert np.array_equal(circuit.encrypt_run_decrypt(x), np.array([[x, x, x], [x, x, x]]))
```

## cnp.one()

Allows you to create encrypted scalar one:

```python
import concrete.numpy as cnp
import numpy as np

@cnp.compiler({"x": "encrypted"})
def f(x):
    z = cnp.one()
    return x + z

inputset = range(10)
circuit = f.compile(inputset)

for x in range(10):
    assert circuit.encrypt_run_decrypt(x) == x + 1
```

## cnp.ones(shape)

Allows you to create encrypted tensor of ones:

```python
import concrete.numpy as cnp
import numpy as np

@cnp.compiler({"x": "encrypted"})
def f(x):
    z = cnp.ones((2, 3))
    return x + z

inputset = range(10)
circuit = f.compile(inputset)

for x in range(10):
    assert np.array_equal(circuit.encrypt_run_decrypt(x), np.array([[x, x, x], [x, x, x]]) + 1)
```

## cnp.univariate(function)

Allows you to wrap any univariate function into a single table lookup:

```python
import concrete.numpy as cnp
import numpy as np

def complex_univariate_function(x):

    def per_element(element):
        result = 0
        for i in range(element):
            result += i
        return result

    return np.vectorize(per_element)(x)

@cnp.compiler({"x": "encrypted"})
def f(x):
    return cnp.univariate(complex_univariate_function)(x)

inputset = [np.random.randint(0, 5, size=(3, 2)) for _ in range(10)]
circuit = f.compile(inputset)

sample = np.array([
    [0, 4],
    [2, 1],
    [3, 0],
])
assert np.array_equal(circuit.encrypt_run_decrypt(sample), complex_univariate_function(sample))
```

{% hint style="danger" %}
The wrapped function shouldn't have any side effects, and it should be deterministic.
{% endhint %}

## coonx.conv(...)

Allows you to perform a convolution operation, with the same semantic of [onnx.Conv](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv):

```python
import concrete.numpy as cnp
import concrete.onnx as connx
import numpy as np

weight = np.array([[2, 1], [3, 2]]).reshape(1, 1, 2, 2)

@cnp.compiler({"x": "encrypted"})
def f(x):
    return connx.conv(x, weight, strides=(2, 2), dilations=(1, 1), group=1)

inputset = [np.random.randint(0, 4, size=(1, 1, 4, 4)) for _ in range(10)]
circuit = f.compile(inputset)

sample = np.array(
    [
        [3, 2, 1, 0],
        [3, 2, 1, 0],
        [3, 2, 1, 0],
        [3, 2, 1, 0],
    ]
).reshape(1, 1, 4, 4)
assert np.array_equal(circuit.encrypt_run_decrypt(sample), f(sample))
```

{% hint style="danger" %}
Only 2D convolutions with one groups and without padding are supported for the time being.
{% endhint %}
