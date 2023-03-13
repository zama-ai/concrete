# Upgrading Guide

## From `Concrete Numpy v0.x` To `Concrete v1`

### The PyPI package `concrete-numpy` is now called `concrete-python`.

### The module `concrete.numpy` is now called `concrete.fhe` and we advise you to use:

```python
from concrete import fhe
```

instead of the previous:

```python
import concrete.numpy as cnp
```

### The module `concrete.onnx` is merged into `concrete.fhe` so we advise you to use:

```python
from concrete import fhe

fhe.conv(...)
fhe.maxpool(...)
```

instead of the previous:

```python
from concrete.onnx import connx

connx.conv(...)
connx.maxpool(...)
```

### Virtual configuration option is removed. Simulation is still supported using the new `simulate` method on circuits:

```python
from concrete import fhe

@fhe.compiler({"x": "encrypted"})
def f(x):
    return x + 42

inputset = range(10)
circuit = f.compile(inputset)

assert circuit.simulate(1) == 43
```

instead of the previous:

```python
import concrete.numpy as cnp

@cnp.compiler({"x": "encrypted"})
def f(x):
    return x + 42

inputset = range(10)
circuit = f.compile(inputset, enable_unsafe_features=True, virtual=True)

assert circuit.encrypt_run_decrypt(1) == 43
```
