# Non-linear operations

In Concrete, there are basically two types of operations:
- linear operations, like additions, subtraction and multiplication by an integer, which are very fast
- and all the rest, which is done by a table lookup (TLU).

TLU are essential to be able to compile all functions, by keeping the semantic of user's program, but
they can be slower, depending on the bitwidth of the inputs of the TLU.

In this document, we explain briefly, from a user point of view, how it works for non-linear operations as comparisons, min/max, bitwise operations, shifts. In [the poweruser documentation](table_lookups_advanced.md), we enter a bit more into the details.

## Changing bit width in the MLIR or dynamically with a TLU

Often, for binary operations, we need to have equivalent bit width for the two operands: it can be done in two ways. Either directly in the MLIR, or dynamically (i.e., at execution time) with a TLU. Because of these different methods, and the fact that none is stricly better than the other one in the general case, we offer different configurations for the non-linear functions.

The first method has the advantage to not require an expensive TLU. However, it may have impact in other parts of the program, since the operand of which we change the bit width may be used elsewhere in the program, so it may create more bit widths changes. Also, if ever the modified operands are used in TLUs, the impact may be significative.

The second method has the advantage to be very local: it has no impact elsewhere. However, it is costly, since it uses a TLU.

## Generic Principle for the user

In the following non-linear operations, we propose a certain number of configurations, using the two methods on the different operands. In general, it is not easy to know in advance which configuration will be the fastest one, but with some Concrete experience. We recommend the users to test and try what are the best configuration depending on their circuits.

By running the following programs with `show_mlir=True`, the advanced user may look the MLIR, and see the different uses of TLUs, bit width changes in the MLIR and dynamic change of the bit width. However, for the classical user, it is not critical to understand the different flavours. We would just recommend to try the different configurations and see which one fits the best for your case.

## Comparisons

For comparison, there are 7 available methods. The generic principle is

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

where `config` is one of
- `fhe.ComparisonStrategy.CHUNKED`
- `fhe.ComparisonStrategy.ONE_TLU_PROMOTED`
- `fhe.ComparisonStrategy.THREE_TLU_CASTED`
- `fhe.ComparisonStrategy.TWO_TLU_BIGGER_PROMOTED_SMALLER_CASTED`
- `fhe.ComparisonStrategy.TWO_TLU_BIGGER_CASTED_SMALLER_PROMOTED`
- `fhe.ComparisonStrategy.THREE_TLU_BIGGER_CLIPPED_SMALLER_CASTED`
- `fhe.ComparisonStrategy.TWO_TLU_BIGGER_CLIPPED_SMALLER_PROMOTED`

## Min / Max operations

For min / max operations, there are 3 available methods. The generic principle is

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

where `config` is one of
- `fhe.MinMaxStrategy.CHUNKED` (default)
- `fhe.MinMaxStrategy.ONE_TLU_PROMOTED`
- `fhe.MinMaxStrategy.THREE_TLU_CASTED`

## Bitwise operations

For bit wise operations (typically, AND, OR, XOR), there are 5 available methods. The generic principle is

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

where `config` is one of
- `fhe.BitwiseStrategy.CHUNKED`
- `fhe.BitwiseStrategy.ONE_TLU_PROMOTED`
- `fhe.BitwiseStrategy.THREE_TLU_CASTED`
- `fhe.BitwiseStrategy.TWO_TLU_BIGGER_PROMOTED_SMALLER_CASTED`
- `fhe.BitwiseStrategy.TWO_TLU_BIGGER_CASTED_SMALLER_PROMOTED`

## Shift operations

For shift operations, there are 2 available methods. The generic principle is

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

where `shifts_with_promotion` is either `True` or `False`.

## Relation with `fhe.multivariate`

Let us just remark that all binary operations described in this document can also be implemented with the `fhe.multivariate` function which is described in [this section](../core-features/extensions.md#fhe.multivariate-function).

```python
import numpy as np
from concrete import fhe


def f(x, y):
    return fhe.multivariate(lambda x, y: x << y)(x, y)


inputset = [(np.random.randint(0, 2**3), np.random.randint(0, 2**2)) for _ in range(100)]

compiler = fhe.Compiler(f, {"x": "encrypted", "y": "encrypted"})
circuit = compiler.compile(inputset, show_mlir=True)
```



