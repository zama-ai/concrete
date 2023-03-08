# Simulation

During development, speed of homomorphic execution is a big blocker for fast prototyping.
You could call the function you're trying to compile directly of course, but it won't be exactly the same as FHE execution, which has a certain probability of error (see [Exactness](../getting-started/exactness.md)).

Considering these, simulation is introduced:

```python
import concrete.numpy as cnp
import numpy as np

@cnp.compiler({"x": "encrypted"})
def f(x):
    return (x + 1) ** 2

inputset = [np.random.randint(0, 10, size=(10,)) for _ in range(10)]
circuit = f.compile(inputset, p_error=0.1)

sample = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

actual = f(sample)
simulation = circuit.simulate(sample)

print(actual.tolist())
print(simulation.tolist())
```

prints

```
[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
[1, 4, 9, 16, 16, 36, 49, 64, 81, 100]
```

{% hint style="warning" %}
Currently, simulation is better than directly calling from Python, but it's not exactly the same with FHE execution. The reason is that it is implemented in Python. Imagine you have an identity table lookup, it might be ommitted from the generated FHE code by the compiler, but it'll be present in Python as optimizations are not done in Python. This will result in a bigger error in simulation. Furthermore, some operations have multiple table lookups within them, and those cannot be simulated unless the actual implementations of said operations are ported to Python. In the future, simulation functionality will be provided by the compiler so all of these issues would be addressed. Until then, keep these in mind.
{% endhint %}
