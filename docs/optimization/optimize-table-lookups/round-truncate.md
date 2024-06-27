### Using round/truncate bit pattern before table lookups

This guide teaches how to improve the execution time of Concrete circuits by using some special operations that reduce the bit width of the input of the table lookup.

There are two extensions which can reduce the bit width of the table lookup input, [fhe.round_bit_pattern(...)](../../core-features/rounding.md) and [fhe.truncate_bit_pattern(...)](../../core-features/truncating.md), which can improve performance by sacrificing exactness.

For example the following code:

```python
import numpy as np
from concrete import fhe

inputset = fhe.inputset(fhe.uint10)
for lsbs_to_remove in range(0, 10):
    def f(x):
        return fhe.round_bit_pattern(x, lsbs_to_remove) // 2

    compiler = fhe.Compiler(f, {"x": "encrypted"})
    circuit = compiler.compile(inputset)

    print(f"{lsbs_to_remove=} -> {int(circuit.complexity):>13_} complexity")
```

prints:

```
lsbs_to_remove=0 -> 9_134_406_574 complexity
lsbs_to_remove=1 -> 3_209_430_092 complexity
lsbs_to_remove=2 -> 1_536_476_735 complexity
lsbs_to_remove=3 -> 1_588_749_586 complexity
lsbs_to_remove=4 ->   848_133_081 complexity
lsbs_to_remove=5 ->   525_987_801 complexity
lsbs_to_remove=6 ->   358_276_023 complexity
lsbs_to_remove=7 ->   373_311_341 complexity
lsbs_to_remove=8 ->   400_596_351 complexity
lsbs_to_remove=9 ->   438_681_996 complexity
```
