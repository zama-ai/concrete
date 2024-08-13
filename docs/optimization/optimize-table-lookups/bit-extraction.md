### Utilizing bit extraction

This guide teaches how to improve the execution time of Concrete circuits by using bit extraction.

[Bit extraction](../../core-features/bit_extraction.md) is a cheap way to extract certain bits of encrypted values. It can be very useful for improving the performance of circuits.

For example:

```python
import numpy as np
from concrete import fhe

inputset = fhe.inputset(fhe.uint6)
for bit_extraction in [False, True]:
    def is_even(x):
        return (
            x % 2 == 0
            if not bit_extraction
            else 1 - fhe.bits(x)[0]
        )

    compiler = fhe.Compiler(is_even, {"x": "encrypted"})
    circuit = compiler.compile(inputset)

    if not bit_extraction:
        print(f"without bit extraction -> {int(circuit.complexity):>11_} complexity")
    else:
        print(f"   with bit extraction -> {int(circuit.complexity):>11_} complexity")
```

prints:

```
without bit extraction -> 230_210_706 complexity
   with bit extraction ->  29_506_014 complexity
```

That's almost 8x improvement to circuit complexity!
