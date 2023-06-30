# Reuse Arguments

Encryption can take quite some time, memory, and network bandwidth if encrypted data is to be transported. Some applications use the same argument, or a set of arguments as one of the inputs. In such applications, it doesn't make sense to encrypt and transfer the arguments each time. Instead, arguments can be encrypted separately, and reused:

```python
from concrete import fhe

@fhe.compiler({"x": "encrypted", "y": "encrypted"})
def add(x, y):
    return x + y

inputset = [(2, 3), (0, 0), (1, 6), (7, 7), (7, 1), (3, 2), (6, 1), (1, 7), (4, 5), (5, 4)]
circuit = add.compile(inputset)

sample_y = 4
_, encrypted_y = circuit.encrypt(None, sample_y)

for sample_x in range(3, 6):
    encrypted_x, _ = circuit.encrypt(sample_x, None)

    encrypted_result = circuit.run(encrypted_x, encrypted_y)
    result = circuit.decrypt(encrypted_result)

    assert result == sample_x + sample_y
```

If you have multiple arguments, the `encrypt` method would return a `tuple`, and if you specify `None` as one of the arguments, `None` is placed at the same location in the resulting `tuple` (e.g., `circuit.encrypt(a, None, b, c, None)` would return `(encrypted_a, None, encrypted_b, encrypted_c, None)`). Each value returned by `encrypt` can be stored and reused anytime.

{% hint style="warning" %}
The ordering of the arguments must be kept consistent! Encrypting an `x` and using it as a `y` could result in undefined behavior.
{% endhint %}
