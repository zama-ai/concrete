# Combining compiled functions with the composable flag

A simple way to say that a function `f` should be compiled such that its outputs can be reused as inputs is to use the
[`composable`](../guides/configure.md#options) configuration setting to `True` when compiling. Doing so, we can then easily compute `f(f(x))` or `f**i(x) = f(f(...(f(x) ..))` for a variable non-encrypted integer `i`, which is typically what happens for recursions.

```python
from concrete import fhe

@fhe.compiler({"counter": "encrypted"})
def increment(counter):
   return (counter + 1) % 100

print("Compiling `increment` function")
increment_fhe = increment.compile(list(range(0, 100)), composable=True)

print("Generating keyset ...")
increment_fhe.keygen()

print("Encrypting the initial counter value")
counter = 0
counter_enc = increment_fhe.encrypt(counter)

print(f"| iteration || decrypted | cleartext |")
for i in range(10):
    counter_enc = increment_fhe.run(counter_enc)
    counter = increment(counter)

    # For demo purpose; no decryption is needed.
    counter_dec = increment_fhe.decrypt(counter_enc)
    print(f"|     {i}     || {counter_dec:<9} | {counter:<9} |")
```

Remark that this option is the equivalent of using the `fhe.AllComposable` policy of [modules](composing_functions_with_modules.md). In particular, the same limitations may occur (see [limitations documentation](composing_functions_with_modules.md#limitations) section).



