### Changing the implementation strategy of complex operations

This guide teaches how to improve the execution time of Concrete circuits by using different conversion strategies for complex operations.

Concrete provides multiple implementation strategies for these complex operations:

- [comparisons (<,<=,==,!=,>=,>)](../../core-features/comparisons.md)
- [bitwise operations (<<,&,|,^,>>)](../../core-features/bitwise.md)
- [minimum and maximum operations](../../core-features/minmax.md)
- [multivariate extension](../../core-features/extensions.md#fhemultivariatefunction)

{% hint style="info" %}
The default strategy is the one that doesn't increase the input bit width, even if it's less optimal than the others. If you don't care about the input bit widths (e.g., if the inputs are only used in this operation), you should definitely change the default strategy.
{% endhint %}

Choosing the correct strategy can lead to big speedups. So if you are not sure which one to use, you can compile with different strategies and compare the complexity.

For example, the following code:

```python
import numpy as np
from concrete import fhe

def f(x, y):
    return x & y

inputset = fhe.inputset(fhe.uint3, fhe.uint4)
strategies = [
    fhe.BitwiseStrategy.ONE_TLU_PROMOTED,
    fhe.BitwiseStrategy.THREE_TLU_CASTED,
    fhe.BitwiseStrategy.TWO_TLU_BIGGER_PROMOTED_SMALLER_CASTED,
    fhe.BitwiseStrategy.TWO_TLU_BIGGER_CASTED_SMALLER_PROMOTED,
    fhe.BitwiseStrategy.CHUNKED,
]

for strategy in strategies:
    compiler = fhe.Compiler(f, {"x": "encrypted", "y": "encrypted"})
    circuit = compiler.compile(inputset, bitwise_strategy_preference=strategy)
    print(
        f"{strategy:>55} "
        f"-> {circuit.programmable_bootstrap_count:>2} TLUs "
        f"-> {int(circuit.complexity):>12_} complexity"
    )
```

prints:

```
                       BitwiseStrategy.ONE_TLU_PROMOTED ->  1 TLUs ->  535_706_740 complexity
                       BitwiseStrategy.THREE_TLU_CASTED ->  3 TLUs ->  599_489_229 complexity
 BitwiseStrategy.TWO_TLU_BIGGER_PROMOTED_SMALLER_CASTED ->  2 TLUs ->  522_239_955 complexity
 BitwiseStrategy.TWO_TLU_BIGGER_CASTED_SMALLER_PROMOTED ->  2 TLUs ->  519_246_216 complexity
                                BitwiseStrategy.CHUNKED ->  6 TLUs ->  358_905_521 complexity
```

or:

```python
import numpy as np
from concrete import fhe

def f(x, y):
    return x == y

inputset = fhe.inputset(fhe.uint4, fhe.uint7)
strategies = [
    fhe.ComparisonStrategy.ONE_TLU_PROMOTED,
    fhe.ComparisonStrategy.THREE_TLU_CASTED,
    fhe.ComparisonStrategy.TWO_TLU_BIGGER_PROMOTED_SMALLER_CASTED,
    fhe.ComparisonStrategy.TWO_TLU_BIGGER_CASTED_SMALLER_PROMOTED,
    fhe.ComparisonStrategy.THREE_TLU_BIGGER_CLIPPED_SMALLER_CASTED,
    fhe.ComparisonStrategy.TWO_TLU_BIGGER_CLIPPED_SMALLER_PROMOTED,
    fhe.ComparisonStrategy.CHUNKED,
]

for strategy in strategies:
    compiler = fhe.Compiler(f, {"x": "encrypted", "y": "encrypted"})
    circuit = compiler.compile(inputset, comparison_strategy_preference=strategy)
    print(
        f"{strategy:>58} "
        f"-> {circuit.programmable_bootstrap_count:>2} TLUs "
        f"-> {int(circuit.complexity):>13_} complexity"
    )
```

prints:

```
                       ComparisonStrategy.ONE_TLU_PROMOTED ->  1 TLUs -> 1_217_510_420 complexity
                       ComparisonStrategy.THREE_TLU_CASTED ->  3 TLUs ->   751_172_128 complexity
 ComparisonStrategy.TWO_TLU_BIGGER_PROMOTED_SMALLER_CASTED ->  2 TLUs -> 1_043_702_103 complexity
 ComparisonStrategy.TWO_TLU_BIGGER_CASTED_SMALLER_PROMOTED ->  2 TLUs -> 1_898_305_707 complexity
ComparisonStrategy.THREE_TLU_BIGGER_CLIPPED_SMALLER_CASTED ->  3 TLUs ->   751_172_128 complexity
ComparisonStrategy.TWO_TLU_BIGGER_CLIPPED_SMALLER_PROMOTED ->  2 TLUs ->   682_694_770 complexity
                                ComparisonStrategy.CHUNKED ->  3 TLUs ->   751_172_128 complexity
```

As you can see, strategies can affect the performance a lot! So make sure to select the appropriate one for your use case if you want to optimize performance.
