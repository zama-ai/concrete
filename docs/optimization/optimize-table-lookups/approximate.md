### Activating approximate mode for rounding

This guide teaches how to improve the execution time of Concrete circuits by using approximate mode for rounding.

You can enable [approximate mode](../../core-features/rounding.md#exactness) to gain even more performance when using rounding by sacrificing some more exactness:

```python
import numpy as np
from concrete import fhe

inputset = fhe.inputset(fhe.uint10)
for lsbs_to_remove in range(0, 10):
    def f(x):
        return fhe.round_bit_pattern(x, lsbs_to_remove, exactness=fhe.Exactness.APPROXIMATE) // 2

    compiler = fhe.Compiler(f, {"x": "encrypted"})
    circuit = compiler.compile(inputset)

    print(f"{lsbs_to_remove=} -> {int(circuit.complexity):>13_} complexity")

```

prints:

```
lsbs_to_remove=0 -> 9_134_406_574 complexity
lsbs_to_remove=1 -> 5_548_275_712 complexity
lsbs_to_remove=2 -> 2_430_793_927 complexity
lsbs_to_remove=3 -> 1_058_638_119 complexity
lsbs_to_remove=4 ->   409_952_712 complexity
lsbs_to_remove=5 ->   172_138_947 complexity
lsbs_to_remove=6 ->    99_198_195 complexity
lsbs_to_remove=7 ->    71_644_380 complexity
lsbs_to_remove=8 ->    55_860_516 complexity
lsbs_to_remove=9 ->    50_978_148 complexity
```
