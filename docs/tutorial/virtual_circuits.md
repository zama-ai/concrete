# Virtual Circuits

During development, speed of homomorphic execution is a big blocker for fast prototyping. Furthermore, it might be desirable to experiment with more bit-widths, even though they are not supported yet, to get insights about the requirements of your system (e.g., we would have an XYZ model with 95% accuracy if we have 25-bits).

To simplify this process, we've introduces virtual circuits:

```python
import concrete.numpy as cnp
import numpy as np

@cnp.compiler({"x": "encrypted"})
def f(x):
    return np.sqrt(x * 100_000).round().astype(np.int64)

inputset = range(100_000, 101_000)
circuit = f.compile(inputset, enable_unsafe_features=True, virtual=True)

print(circuit)
print(circuit.encrypt_run_decrypt(100_500), "~=", np.sqrt(100_500 * 100_000))
```

prints

```
%0 = x                       # EncryptedScalar<uint17>        ∈ [100000, 100999]
%1 = 100000                  # ClearScalar<uint17>            ∈ [100000, 100000]
%2 = multiply(%0, %1)        # EncryptedScalar<uint34>        ∈ [10000000000, 10099900000]
%3 = subgraph(%2)            # EncryptedScalar<uint17>        ∈ [100000, 100498]
return %3

Subgraphs:

    %3 = subgraph(%2):

        %0 = input                         # EncryptedScalar<uint1>
        %1 = sqrt(%0)                      # EncryptedScalar<float64>
        %2 = around(%1, decimals=0)        # EncryptedScalar<float64>
        %3 = astype(%2, dtype=int_)        # EncryptedScalar<uint1>
        return %3
        
100250 ~= 100249.6882788171
```

and it doesn't perform any homomorphic computation. It just simulates execution.

Keyword arguments `enable_unsafe_features=True` and `virtual=True` passed to `compile` are configuration options. `virtaul=True` enables makes the circuit virtual, and because virtual circuits are highly experimental, unsafe features must be enabled using `enable_unsafe_features=True` to utilize virtual circuits. See [How to Configure](../howto/configure.md) to learn more about configuration options.

{% hint style="info" %}
Virtual circuits still check for operational constraints and type constraints. Which means you cannot have floating points, or unsupported operations. They just ignore bit-width constraints.
{% endhint %}

{% hint style="warning" %}
Virtual circuits are still experimental, and they don't properly consider [error probability](../getting-started/exactness.md) for example. That's why you need to enable unsafe features to use them. Use them with care!
{% endhint %}
