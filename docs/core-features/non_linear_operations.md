# Non-linear operations
This document introduces the usages and optimization strategies of non-linear operations in **Concrete**, focusing on comparisons, min/max operations, bitwise operations, and shifts. For a more in-depth explanation on advanced options, refer to the [Table Lookup advanced documentation](table_lookups_advanced.md).

## Overview of non-linear operations

In **Concrete**, there are two types of operations:
- **Linear operations**: These include additions, subtractions, and multiplications by an integer. They are computationally fast.
- **Non-linear operations**: These require [Table Lookups (TLUs)](../core-features/table_lookups.md) to maintain the semantic integrity of the user's program. The performance of TLUs is slower and vary depending on the bit width of the inputs.


## Changing bit width in the MLIR or dynamically with a TLU

Binary operations often require operands to have matching bit widths. This adjustment can be achieved in two ways: either directly within the MLIR or dynamically at execution time using a TLU. Each method has its own advantages and trade-offs, so Concrete provides multiple configuration options for non-linear functions.

**MLIR adjustment:** This method doesn't require an expensive TLU. However, it may affect other parts of your program if the adjusted operand is used elsewhere, potentially causing more changes.

**Dynamic adjustment with TLU:** This method is more localized and won’t impact other parts of your program, but it’s more expensive due to the cost of using a TLU.

## General guidelines

In the following non-linear operations, we propose a certain number of configurations, using the two methods on the different operands. It’s not always clear which option will be the fastest, so we recommend trying out different configurations to see what works best for your circuit.

Note that you have the option to set `show_mlir=True` to view how the MLIR handles TLUs and bit width changes. However, it's not essential to understand these details. So we recommend just testing the configurations and pick the one that performs best for your case.

## Comparisons

For comparison, there are 7 available methods. Here's the general principle:

```python
import numpy as np
from concrete import fhe

configuration = fhe.Configuration(
    comparison_strategy_preference=config,
)

def f(x, y):
    return x < y

inputset = [
    (np.random.randint(0, 2**4), np.random.randint(0, 2**4))
    for _ in range(100)
]

compiler = fhe.Compiler(f, {"x": "encrypted", "y": "encrypted"})
circuit = compiler.compile(inputset, configuration, show_mlir=True)
```

The `config` can be one of the following:
- `fhe.ComparisonStrategy.CHUNKED`
- `fhe.ComparisonStrategy.ONE_TLU_PROMOTED`
- `fhe.ComparisonStrategy.THREE_TLU_CASTED`
- `fhe.ComparisonStrategy.TWO_TLU_BIGGER_PROMOTED_SMALLER_CASTED`
- `fhe.ComparisonStrategy.TWO_TLU_BIGGER_CASTED_SMALLER_PROMOTED`
- `fhe.ComparisonStrategy.THREE_TLU_BIGGER_CLIPPED_SMALLER_CASTED`
- `fhe.ComparisonStrategy.TWO_TLU_BIGGER_CLIPPED_SMALLER_PROMOTED`

## Min / Max operations

For min / max operations, there are 3 available methods. Here's the general principle:

```python
import numpy as np
from concrete import fhe

configuration = fhe.Configuration(
    min_max_strategy_preference=config,
)

def f(x, y):
    return np.minimum(x, y)

inputset = [
    (np.random.randint(0, 2**4), np.random.randint(0, 2**2))
    for _ in range(100)
]

compiler = fhe.Compiler(f, {"x": "encrypted", "y": "encrypted"})
circuit = compiler.compile(inputset, configuration, show_mlir=True)
```

The `config` can be one of the following:
- `fhe.MinMaxStrategy.CHUNKED` (default)
- `fhe.MinMaxStrategy.ONE_TLU_PROMOTED`
- `fhe.MinMaxStrategy.THREE_TLU_CASTED`

## Bitwise operations

For bit wise operations (typically, AND, OR, XOR), there are 5 available methods. Here's the general principle:

```python
import numpy as np
from concrete import fhe

configuration = fhe.Configuration(
    bitwise_strategy_preference=config,
)

def f(x, y):
    return x & y

inputset = [
    (np.random.randint(0, 2**4), np.random.randint(0, 2**4))
    for _ in range(100)
]

compiler = fhe.Compiler(f, {"x": "encrypted", "y": "encrypted"})
circuit = compiler.compile(inputset, configuration, show_mlir=True)
```

The `config` can be one of the following:
- `fhe.BitwiseStrategy.CHUNKED`
- `fhe.BitwiseStrategy.ONE_TLU_PROMOTED`
- `fhe.BitwiseStrategy.THREE_TLU_CASTED`
- `fhe.BitwiseStrategy.TWO_TLU_BIGGER_PROMOTED_SMALLER_CASTED`
- `fhe.BitwiseStrategy.TWO_TLU_BIGGER_CASTED_SMALLER_PROMOTED`

## Shift operations

For shift operations, there are 2 available methods. Here's the general principle:

```python
import numpy as np
from concrete import fhe

configuration = fhe.Configuration(
    shifts_with_promotion=shifts_with_promotion,
)

def f(x, y):
    return x << y

inputset = [
    (np.random.randint(0, 2**3), np.random.randint(0, 2**2))
    for _ in range(100)
]

compiler = fhe.Compiler(f, {"x": "encrypted", "y": "encrypted"})
circuit = compiler.compile(inputset, configuration, show_mlir=True)
```

The `shifts_with_promotion` is either `True` or `False`.

## Relation with `fhe.multivariate`

All binary operations described in this document can also be implemented with the `fhe.multivariate` function which is described in [ fhe.multivariate function documentation](../core-features/extensions.md#fhe.multivariate-function). Here's an example:

```python
import numpy as np
from concrete import fhe


def f(x, y):
    return fhe.multivariate(lambda x, y: x << y)(x, y)


inputset = [(np.random.randint(0, 2**3), np.random.randint(0, 2**2)) for _ in range(100)]

compiler = fhe.Compiler(f, {"x": "encrypted", "y": "encrypted"})
circuit = compiler.compile(inputset, show_mlir=True)
```



