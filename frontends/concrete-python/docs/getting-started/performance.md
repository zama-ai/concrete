# Performance

The most important operation in Concrete-Numpy is the table lookup operation. All operations except addition, subtraction, multiplication with non-encrypted values, and a few operations built with those primitive operations (e.g. matmul, conv) are converted to table lookups under the hood:

```python
import concrete.numpy as cnp

@cnp.compiler({"x": "encrypted"})
def f(x):
    return x ** 2

inputset = range(2 ** 4)
circuit = f.compile(inputset)
```

is exactly the same as

```python
import concrete.numpy as cnp

table = cnp.LookupTable([x ** 2 for x in range(2 ** 4)])

@cnp.compiler({"x": "encrypted"})
def f(x):
    return table[x]

inputset = range(2 ** 4)
circuit = f.compile(inputset)
```

Table lookups are very flexible, and they allow Concrete Numpy to support many operations, but they are expensive! Therefore, you should try to avoid them as much as possible. In most cases, it's not possible to avoid them completely, but you might remove the number of TLUs or replace some of them with other primitive operations.

The exact cost depend on many variables (machine configuration, error probability, etc.), but you can develop some intuition for single threaded CPU execution performance using:

```python
import time

import concrete.numpy as cnp
import numpy as np

WARMUP = 3
SAMPLES = 8
BITWIDTHS = range(1, 15)
CONFIGURATION = cnp.Configuration(
    enable_unsafe_features=True,
    use_insecure_key_cache=True,
    insecure_key_cache_location=".keys",
)

timings = {}
for n in BITWIDTHS:
    @cnp.compiler({"x": "encrypted"})
    def base(x):
        return x

    table = cnp.LookupTable([np.sqrt(x).round().astype(np.int64) for x in range(2 ** n)])

    @cnp.compiler({"x": "encrypted"})
    def tlu(x):
        return table[x]

    inputset = [0, 2**n - 1]

    base_circuit = base.compile(inputset, CONFIGURATION)
    tlu_circuit = tlu.compile(inputset, CONFIGURATION)

    print()
    print(f"Generating keys for n={n}...")

    base_circuit.keygen()
    tlu_circuit.keygen()

    timings[n] = []
    for i in range(SAMPLES + WARMUP):
        sample = np.random.randint(0, 2 ** n)

        encrypted_sample = base_circuit.encrypt(sample)
        start = time.time()
        encrypted_result = base_circuit.run(encrypted_sample)
        end = time.time()
        assert base_circuit.decrypt(encrypted_result) == sample

        base_time = end - start

        encrypted_sample = tlu_circuit.encrypt(sample)
        start = time.time()
        encrypted_result = tlu_circuit.run(encrypted_sample)
        end = time.time()
        assert tlu_circuit.decrypt(encrypted_result) == np.sqrt(sample).round().astype(np.int64)

        tlu_time = end - start

        if i >= WARMUP:
            timings[n].append(tlu_time - base_time)
            print(f"Sample #{i - WARMUP + 1} took {timings[n][-1] * 1000:.3f}ms")

print()
for n, times in timings.items():
    print(f"{n}-bits -> {np.mean(times) * 1000:.3f}ms")
```

{% hint style="info" %}
Concrete Numpy automatically parallelize execution if TLUs are applied to tensors.
{% endhint %}
