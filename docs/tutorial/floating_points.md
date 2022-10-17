# Floating Points

**Concrete-Numpy** partly supports floating points:

* They cannot be inputs
* They cannot be outputs
* They can be intermediate values under certain constraints

## As intermediate values

**Concrete-Compile**, which is used for compiling the circuit, doesn't support floating points at all. However, it supports table lookups. They take an integer and map it to another integer. It does not care how the lookup table is calculated. Further, the constraints of this operation are such that there should be a single integer input and it should result in a single integer output.

As long as your floating point operations comply with those constraints, **Concrete-Numpy** automatically converts your operations to a table lookup operation:

```python
import concrete.numpy as cnp
import numpy as np

@cnp.compiler({"x": "encrypted"})
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

In the example above, `a`, `b`, and `c` are all floating point intermediates. However, they are just used to calculate `d`, which is an integer and value of `d` dependent upon `x` , which is another integer. **Concrete-Numpy** detects this and fuses all of those operations into a single table lookup from `x` to `d`.

This approach works for a variety of use cases, but it comes up short for some:

<!--pytest-codeblocks:skip-->
```python
import concrete.numpy as cnp
import numpy as np

@cnp.compiler({"x": "encrypted", "y": "encrypted"})
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

... results in:

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

The reason for this is that `d` no longer depends solely on `x`, it depends on `y` as well. **Concrete-Numpy** cannot fuse these operations, so it raises an exception instead.
