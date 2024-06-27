### Reducing the amount of table lookups

This guide teaches how to improve the execution time of Concrete circuits by reducing the amount of table lookups.

Reducing the amount of table lookups is probably the most complicated guide in this section as it's not automated. The idea is to use mathematical properties of operations to reduce the amount of table lookups needed to achieve the result.

One great example is in adding big integers in bitmap representation. Here is the basic implementation:

```python
def add_bitmaps(x, y):
    result = fhe.zeros((N,))
    carry = 0

    addition = x + y
    for i in range(N):
        addition_and_carry = addition[i] + carry
        carry = addition_and_carry >> 1
        result[i] = addition_and_carry % 2

    return result
```

There are two table lookups within the loop body, one for `>>` and one for `%`.

This implementation is not optimal though, since the same output can be achieved with just a single table lookup:

```python
def add_bitmaps(x, y):
    result = fhe.zeros((N,))
    carry = 0

    addition = x + y
    for i in range(N):
        addition_and_carry = addition[i] + carry
        carry = addition_and_carry >> 1
        result[i] = addition_and_carry - (carry * 2)

    return result
```

It was possible to do this because the original operations had a mathematical equivalence with the optimized operations and optimized operations achieved the same output with less table lookups!

Here is the full code example and some numbers for this optimization:

```python
import numpy as np
from concrete import fhe

N = 32

def add_bitmaps_naive(x, y):
    result = fhe.zeros((N,))
    carry = 0

    addition = x + y
    for i in range(N):
        addition_and_carry = addition[i] + carry
        carry = addition_and_carry >= 2
        result[i] = addition_and_carry % 2

    return result

def add_bitmaps_optimized(x, y):
    result = fhe.zeros((N,))
    carry = 0

    addition = x + y
    for i in range(N):
        addition_and_carry = addition[i] + carry
        carry = addition_and_carry >> 1
        result[i] = addition_and_carry - (carry * 2)

    return result

inputset = fhe.inputset(fhe.tensor[fhe.uint1, N], fhe.tensor[fhe.uint1, N])
for (name, implementation) in [("naive", add_bitmaps_naive), ("optimized", add_bitmaps_optimized)]:
    compiler = fhe.Compiler(implementation, {"x": "encrypted", "y": "encrypted"})
    circuit = compiler.compile(inputset)

    print(
        f"{name:>9} implementation "
        f"-> {int(circuit.programmable_bootstrap_count)} table lookups "
        f"-> {int(circuit.complexity):_} complexity"
    )
```

prints:

```
    naive implementation -> 63 table lookups -> 2_427_170_697 complexity
optimized implementation -> 32 table lookups -> 1_224_206_208 complexity
```

which is almost half the amount of table lookups and ~2x less complexity for the same operation!
