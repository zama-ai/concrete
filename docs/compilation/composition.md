# Combining compiled functions with the composable flag

This document explains how to combine compiled functions with the `composable` flag in **Concrete**. 

By setting the `composable` flag to `True`, you can compile a function such that its outputs can be reused as inputs. For example, you can then easily compute `f(f(x))` or `f**i(x) = f(f(...(f(x) ..))` for a non-encrypted integer `i` variable, which is usually required for recursions.

Here is an example:

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

Remark that this option is the equivalent to using the `fhe.AllComposable` policy of [modules](composing_functions_with_modules.md). In particular, the same limitations may occur (see [limitations documentation](composing_functions_with_modules.md#limitations) section).



