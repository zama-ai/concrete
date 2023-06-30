# Decorator

If you are trying to compile a regular function, you can use the decorator interface instead of the explicit `Compiler` interface to simplify your code:

```python
from concrete import fhe

@fhe.compiler({"x": "encrypted"})
def f(x):
    return x + 42

inputset = range(10)
circuit = f.compile(inputset)

assert circuit.encrypt_run_decrypt(10) == f(10)
```

{% hint style="info" %}
This decorator is a way to add the `compile` method to the function object without changing its name elsewhere.
{% endhint %}
