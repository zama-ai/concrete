# Extensions

**Concrete** supports native Python and NumPy operations as much as possible, but not everything is available in Python or NumPy. So, we provide some extensions ourselves to improve your experience.

## fhe.univariate(function)

Allows you to wrap any univariate function into a single table lookup:

```python
import numpy as np
from concrete import fhe

def complex_univariate_function(x):

    def per_element(element):
        result = 0
        for i in range(element):
            result += i
        return result

    return np.vectorize(per_element)(x)

@fhe.compiler({"x": "encrypted"})
def f(x):
    return fhe.univariate(complex_univariate_function)(x)

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
The wrapped function shouldn't have any side effects, and it should be deterministic. Otherwise, the outcome is undefined.
{% endhint %}

## fhe.conv(...)

Allows you to perform a convolution operation, with the same semantic of [onnx.Conv](https://github.com/onnx/onnx/blob/main/docs/Operators.md#conv):

```python
import numpy as np
from concrete import fhe

weight = np.array([[2, 1], [3, 2]]).reshape(1, 1, 2, 2)

@fhe.compiler({"x": "encrypted"})
def f(x):
    return fhe.conv(x, weight, strides=(2, 2), dilations=(1, 1), group=1)

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
Only 2D convolutions without padding and with one groups are supported for the time being.
{% endhint %}

## fhe.maxpool(...)

Allows you to perform a maxpool operation, with the same semantic of [onnx.MaxPool](https://github.com/onnx/onnx/blob/main/docs/Operators.md#maxpool):

```python
import numpy as np
from concrete import fhe

@fhe.compiler({"x": "encrypted"})
def f(x):
    return fhe.maxpool(x, kernel_shape=(2, 2), strides=(2, 2), dilations=(1, 1))

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
Only 2D maxpooling without padding up to 15-bits is supported for the time being.
{% endhint %}

## fhe.array(...)

Allows you to create encrypted arrays:

```python
import numpy as np
from concrete import fhe

@fhe.compiler({"x": "encrypted", "y": "encrypted"})
def f(x, y):
    return fhe.array([x, y])

inputset = [(3, 2), (7, 0), (0, 7), (4, 2)]
circuit = f.compile(inputset)

sample = (3, 4)
assert np.array_equal(circuit.encrypt_run_decrypt(*sample), f(*sample))
```

{% hint style="danger" %}
Only scalars can be used to create arrays for the time being.
{% endhint %}

## fhe.zero()

Allows you to create encrypted scalar zero:

```python
from concrete import fhe
import numpy as np

@fhe.compiler({"x": "encrypted"})
def f(x):
    z = fhe.zero()
    return x + z

inputset = range(10)
circuit = f.compile(inputset)

for x in range(10):
    assert circuit.encrypt_run_decrypt(x) == x
```

## fhe.zeros(shape)

Allows you to create encrypted tensor of zeros:

```python
from concrete import fhe
import numpy as np

@fhe.compiler({"x": "encrypted"})
def f(x):
    z = fhe.zeros((2, 3))
    return x + z

inputset = range(10)
circuit = f.compile(inputset)

for x in range(10):
    assert np.array_equal(circuit.encrypt_run_decrypt(x), np.array([[x, x, x], [x, x, x]]))
```

## fhe.one()

Allows you to create encrypted scalar one:

```python
from concrete import fhe
import numpy as np

@fhe.compiler({"x": "encrypted"})
def f(x):
    z = fhe.one()
    return x + z

inputset = range(10)
circuit = f.compile(inputset)

for x in range(10):
    assert circuit.encrypt_run_decrypt(x) == x + 1
```

## fhe.ones(shape)

Allows you to create encrypted tensor of ones:

```python
from concrete import fhe
import numpy as np

@fhe.compiler({"x": "encrypted"})
def f(x):
    z = fhe.ones((2, 3))
    return x + z

inputset = range(10)
circuit = f.compile(inputset)

for x in range(10):
    assert np.array_equal(circuit.encrypt_run_decrypt(x), np.array([[x, x, x], [x, x, x]]) + 1)
```
