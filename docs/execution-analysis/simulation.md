# Simulation
This document explains how to use simulation to speed up the development, enabling faster prototyping while accounting for the inherent probability of errors in Fully Homomorphic Encryption (FHE) execution. 

## Using simulation for faster prototyping
During development, the speed of homomorphic execution can be a blocker for fast prototyping. Although you can directly call the function you want to compile, this approach does not fully replicate FHE execution, which involves a certain probability of error (see [Exactness](../core-features/table\_lookups.md#table-lookup-exactness)).

To overcome this issue, simulation is introduced:

```python
from concrete import fhe
import numpy as np

@fhe.compiler({"x": "encrypted"})
def f(x):
    return (x + 1) ** 2

inputset = [np.random.randint(0, 10, size=(10,)) for _ in range(10)]
circuit = f.compile(inputset, p_error=0.1, fhe_simulation=True)

sample = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

actual = f(sample)
simulation = circuit.simulate(sample)

print(actual.tolist())
print(simulation.tolist())
```

After the simulation runs, it prints the following results:

```
[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
[1, 4, 9, 16, 16, 36, 49, 64, 81, 100]
```

## Overflow detection in simulation

Overflow can happen during an FHE computation, leading to unexpected behaviors. Using simulation can help you detect these events by printing a warning whenever an overflow happens. This feature is disabled by default, but you can enable it by setting `detect_overflow_in_simulation=True` during compilation.

To demonstrate, we will compile the previous circuit with overflow detection enabled and trigger an overflow:

```python
# compile with overflow detection enabled
circuit = f.compile(inputset, p_error=0.1, fhe_simulation=True, detect_overflow_in_simulation=True)
# cause an overflow
circuit.simulate([0,1,2,3,4,5,6,7,8,15])
```

You will see the following warning after the simulation call:

```bash
WARNING at loc("script.py":3:0): overflow happened during addition in simulation
```

If you look at the MLIR (`circuit.mlir`), you will see that the input type is supposed to be `eint4` represented in 4 bits with a maximum value of 15. Since there's an addition of the input, we used the maximum value (15) here to trigger an overflow (15 + 1 = 16 which needs 5 bits). The warning specifies the operation that caused the overflow and its location. Similar warnings will be displayed for all basic FHE operations such as add, mul, and lookup tables.
