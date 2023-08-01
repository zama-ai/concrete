# Simulation

During development, the speed of homomorphic execution can be a blocker for fast prototyping. You could call the function you're trying to compile directly, of course, but it won't be exactly the same as FHE execution, which has a certain probability of error (see [Exactness](../getting-started/exactness.md)).

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

After the simulation runs, it prints the following:

```
[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
[1, 4, 9, 16, 16, 36, 49, 64, 81, 100]
```

{% hint style="warning" %}
Currently, simulation is better than directly calling from Python, but it's not exactly the same as FHE execution. This is because it is implemented in Python.&#x20;

Imagine you have an identity table lookup. In the FHE code, this may be omitted by the Compiler. In the Python simulation, it would still be present as the optimizations used in the Compiler are not considered. This will result in a bigger error in simulation.&#x20;

Some operations are made up of multiple table lookups, and operations of this type cannot be simulated unless their implementation is ported to Python. In the future, simulation functionality will be provided by the Compiler, so all of these issues will be addressed.
{% endhint %}
