# Floating Points

This document describes how floating points are treated and manipulated in Concrete.

**Concrete** partly supports floating points. There is no support for floating point inputs or outputs. However, there is support for intermediate values to be floating points (under certain constraints). Also, we note that one can use an equivalent of fixed points in Concrete, as described in [our tutorial](../../frontends/concrete-python/examples/floating_point/floating_point.ipynb).

## Floating points as intermediate values

**Concrete-Compile**, which is used for compiling the circuit, doesn't support floating points at all. However, it supports table lookups which take an integer and map it to another integer. The constraints of this operation are that there should be a single integer input, and a single integer output.

As long as your floating point operations comply with those constraints, **Concrete** automatically converts them to a table lookup operation:

```python
from concrete import fhe
import numpy as np

@fhe.compiler({"x": "encrypted"})
def f(x):
    a = x + 1.5
    b = np.sin(x)
    c = np.around(a + b)
    d = c.astype(np.int64)
    return d

inputset = range(8)
circuit = f.compile(inputset)

for x in range(8):
    assert circuit.encrypt_run_decrypt(x) == f(x)
```

In the example above, `a`, `b`, and `c` are floating point intermediates. They are used to calculate `d`, which is an integer with a value dependent upon `x`, which is also an integer. **Concrete** detects this and fuses all of these operations into a single table lookup from `x` to `d`.

This approach works for a variety of use cases, but it comes up short for others:

<!--pytest-codeblocks:skip-->
```python
from concrete import fhe
import numpy as np

@fhe.compiler({"x": "encrypted", "y": "encrypted"})
def f(x, y):
    a = x + 1.5
    b = np.sin(y)
    c = np.around(a + b)
    d = c.astype(np.int64)
    return d

inputset = [(1, 2), (3, 0), (2, 2), (1, 3)]
circuit = f.compile(inputset)

for x in range(8):
    assert circuit.encrypt_run_decrypt(x) == f(x)
```

This results in:

```
RuntimeError: Function you are trying to compile cannot be converted to MLIR

%0 = x                             # EncryptedScalar<uint2>
%1 = 1.5                           # ClearScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only integer constants are supported
%2 = y                             # EncryptedScalar<uint2>
%3 = add(%0, %1)                   # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only integer operations are supported
%4 = sin(%2)                       # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only integer operations are supported
%5 = add(%3, %4)                   # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only integer operations are supported
%6 = around(%5)                    # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only integer operations are supported
%7 = astype(%6, dtype=int_)        # EncryptedScalar<uint3>
return %7
```

The reason for the error is that `d` no longer depends solely on `x`; it depends on `y` as well. **Concrete** cannot fuse these operations, so it raises an exception instead.
