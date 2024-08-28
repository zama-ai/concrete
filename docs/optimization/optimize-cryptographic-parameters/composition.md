### Specifying composition when using modules

This guide explains how to optimize cryptographic parameters by specifying composition when using [modules](../../compilation/composing_functions_with_modules.md).

When using [modules](../../compilation/composing_functions_with_modules.md), make sure to specify [composition](../../compilation/composing_functions_with_modules.md#optimizing-runtimes-with-composition-policies) so that the compiler can select more optimal parameters based on how the functions in the module would be used.

For example:

```python
import numpy as np
from concrete import fhe


@fhe.module()
class PowerWithoutComposition:
    @fhe.function({"x": "encrypted"})
    def square(x):
        return x ** 2

    @fhe.function({"x": "encrypted"})
    def cube(x):
        return x ** 3

without_composition = PowerWithoutComposition.compile(
    {
        "square": fhe.inputset(fhe.uint2),
        "cube": fhe.inputset(fhe.uint4),
    }
)
print(f"without composition -> {int(without_composition.complexity):>10_} complexity")


@fhe.module()
class PowerWithComposition:
    @fhe.function({"x": "encrypted"})
    def square(x):
        return x ** 2

    @fhe.function({"x": "encrypted"})
    def cube(x):
        return x ** 3

    composition = fhe.Wired(
        [
            fhe.Wire(fhe.Output(square, 0), fhe.Input(cube, 0))
        ]
    )

with_composition = PowerWithComposition.compile(
    {
        "square": fhe.inputset(fhe.uint2),
        "cube": fhe.inputset(fhe.uint4),
    }
)
print(f"   with composition -> {int(with_composition.complexity):>10_} complexity")
```

This prints:

```
without composition -> 185_863_835 complexity
   with composition -> 135_871_612 complexity
```

It means that specifying composition resulted in ~35% improvement to complexity for computing `cube(square(x))`.
