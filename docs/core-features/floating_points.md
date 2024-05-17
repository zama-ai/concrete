# Floating points

This document explains how **Concrete** handles floating point operations, with examples and limitations.

## Introduction

**Concrete** partly supports floating points. While it doesn't support floating point inputs or outputs, it does allow floating points as intermediate values under certain constraints.

The tool to compile circuits - `Concrete-Compile` doesn't support floating points directly. However, it supports Table Lookups (TLUs) which take an integer and map it to another integer. The constraint of this operation is that the input and output must both be single integers.

If your floating point operations comply with the constraint, **Concrete** automatically converts them to TLU operations.

## How to use in FHE

In the following example, `a`, `b`, and `c` are floating-point intermediates used to calculate an integer `d`, which is dependent on another integer `x`. **Concrete** detects this and fuses all of these operations into a single TLU from `x` to `d`:

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

## Limitation

This approach works for many use cases, but not for all. In the following example, `d` depends not only on `x` but on both `x` and `y`. **Concrete** cannot fuse these operations, so it raises an error.

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
