# Upgrading Guide

## From `Concrete v1` To `Concrete v2`

### Encrypt/Run/Decrypt type signatures are changed

In `Concrete v1`, there were `PublicArguments` and `PublicResult` types. `PublicArguments` was used to encapsulate encrypted and clear arguments to the function and `PublicResult` was used to store the encrypted result of the function.

```python
x, y, z = 10, 20, 30
encrypted_args = circuit.encrypt(x, y, z)
encrypted_result = circuit.run(encrypted_args)
result = circuit.decrypt(encrypted_result)
```

where `encrypted_args` is of type `fhe.PublicArguments`, and `encrypted_result` is of type `fhe.PublicResult`.

This was simple, but limiting (e.g., since `PublicArguments` contained all arguments, a change in one of the arguments required re-encryption of all arguments).

In `Concrete v2`, there is `fhe.Value` type, which is the result of all of encrypt, run, and decrypt.

```python
x, y, z = 10, 20, 30
encrypted_x, encrypted_y, encrypted_z = circuit.encrypt(x, y, z)
encrypted_result = circuit.run(encrypted_x, encrypted_y, encrypted_z)
result = circuit.decrypt(encrypted_result)
```

where all of `encrypted_x`, `encrypted_y`, `encrypted_z` and `encrypted_result` are `fhe.Value`.

In case `y` value needs to be changed whereas the others need to remain, it's possible to do:

```python
new_y = 42
_, new_encrypted_y, _ = circuit.encrypt(None, new_y, None)

new_encrypted_result = circuit.run(encrypted_x, new_encrypted_y, encrypted_z)
new_result = circuit.decrypt(new_encrypted_result)
```

New `fhe.Circuit.run(...)` also support tuples so the old code still works! However, `fhe.Server.run(...)` had another parameter called `evaluation_keys`, which have to be a keyword argument now as function arguments are variadic now!

```python
server: fhe.Server = ...
encrypted_x: fhe.Value = ...
encrypted_y: fhe.Value = ...
encrypted_z: fhe.Value = ...
evaluation_keys: fhe.EvaluationKeys = ...

server.run(encrypted_x, encrypted_y, encrypted_z, evaluation_keys=evaluation_keys)
                                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                                  this needs to be a keyword argument now
```

Whereas previously it could be:

```python
server: fhe.Server = ...
encrypted_args: fhe.PublicArguments = ...
evaluation_keys: fhe.EvaluationKeys = ...

server.run(encrypted_args, evaluation_keys)
```

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
