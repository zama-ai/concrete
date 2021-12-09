# Working With Floating Points

## An example

```python
import numpy as np
import concrete.numpy as hnp

# Function using floating points values converted back to integers at the end
def f(x):
    return np.fabs(50 * (2 * np.sin(x) * np.cos(x))).astype(np.uint32)
    # astype is to go back to the integer world

# Compiling with x encrypted
compiler = hnp.NPFHECompiler(f, {"x": "encrypted"})
compiler.eval_on_inputset(range(64))

circuit = compiler.get_compiled_fhe_circuit()

assert circuit.run(3) == f(3)
assert circuit.run(0) == f(0)
assert circuit.run(1) == f(1)
assert circuit.run(10) == f(10)
assert circuit.run(60) == f(60)

print("All good!")
```

One can look to [numpy supported functions](../howto/NUMPY_SUPPORT.md) for information about possible float operations.


## Limitations

Floating point support in **Concrete** is very limited for the time being. They can't appear on inputs, or they can't be outputs. However, they can be used in intermediate results. Unfortunately, there are limitations on that front as well.

This biggest one is that, because floating point operations are fused into table lookups with a single unsigned integer input and single unsigned integer output, only univariate portion of code can be replaced with table lookups, which means multivariate portions cannot be compiled.

To give a precise example, `100 - np.fabs(50 * (np.sin(x) + np.sin(y)))` cannot be compiled because the floating point part depends on both `x` and `y` (i.e., it cannot be rewritten in the form `100 - table[z]` for a `z` that could be computed easily from `x` and `y`).

To dive into implementation details, you may refer to [Fusing Floating Point Operations](../../dev/explanation/FLOAT-FUSING.md) document.
