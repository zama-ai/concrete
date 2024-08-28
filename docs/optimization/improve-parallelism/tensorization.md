### Tensorizing operations

This guide explains tensorization and how it can improve the execution time of **Concrete** circuits.

Tensors should be used instead of scalars when possible to maximize loop parallelism.

For example:

```python
import time

import numpy as np
from concrete import fhe

inputset = fhe.inputset(fhe.uint6, fhe.uint6, fhe.uint6)
for tensorize in [False, True]:
    def f(x, y, z):
        return (
            np.sum(fhe.array([x, y, z]) ** 2)
            if tensorize
            else (x ** 2) + (y ** 2) + (z ** 2)
        )

    compiler = fhe.Compiler(f, {"x": "encrypted", "y": "encrypted", "z": "encrypted"})
    circuit = compiler.compile(inputset)

    circuit.keygen()
    for sample in inputset[:3]:  # warmup
        circuit.encrypt_run_decrypt(*sample)

    timings = []
    for sample in inputset[3:13]:
        start = time.time()
        result = circuit.encrypt_run_decrypt(*sample)
        end = time.time()

        assert np.array_equal(result, f(*sample))
        timings.append(end - start)

    if not tensorize:
        print(f"without tensorization -> {np.mean(timings):.03f}s")
    else:
        print(f"   with tensorization -> {np.mean(timings):.03f}s")
```

This prints:

```
without tensorization -> 0.214s
   with tensorization -> 0.118s
```

{% hint style="info" %}
Enabling dataflow is kind of letting the runtime do this for you. It'd also help in the specific case.
{% endhint %}
