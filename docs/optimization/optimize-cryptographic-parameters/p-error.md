### Adjusting table lookup error probability

This guide explains how setting `p_error` configuration option can affect the performance of **Concrete** circuits.

Adjusting table lookup error probability is discussed extensively in [Table lookup exactness](../../core-features/table_lookups_advanced.md#table-lookup-exactness) section. The idea is to sacrifice exactness to gain performance.

For example:

```python
import numpy as np
from concrete import fhe

def f(x, y):
    return (x // 2) * (y // 3)

inputset = fhe.inputset(fhe.uint4, fhe.uint4)
for p_error in [(1 / 1_000_000), (1 / 100_000), (1 / 10_000), (1 / 1_000), (1 / 100)]:
    compiler = fhe.Compiler(f, {"x": "encrypted", "y": "encrypted"})
    circuit = compiler.compile(inputset, p_error=p_error)
    print(f"p_error of {p_error:.6f} -> {int(circuit.complexity):_} complexity")
```

This prints:

```
p_error of 0.000001 -> 294_773_524 complexity
p_error of 0.000010 -> 286_577_520 complexity
p_error of 0.000100 -> 275_887_080 complexity
p_error of 0.001000 -> 265_196_640 complexity
p_error of 0.010000 -> 184_144_972 complexity
```
