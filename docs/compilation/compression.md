# Compression
This document explains the compression feature in **Concrete** and its performance impact.

Fully Homomorphic Encryption (FHE) needs both ciphertexts (encrypted data) and evaluation keys to carry out the homomorphic evaluation of a function. Both elements are large, which may critically affect the application's performance depending on the use case, application deployment, and the method for transmitting and storing ciphertexts and evaluation keys.

## Enabling compression
During compilation, you can enable compression options to enforce the use of compression features. The two available compression options are:

* **compress\_evaluation\_keys**: bool = False,
    - This specifies that serialization takes the compressed form of evaluation keys.
* **compress\_input\_ciphertexts**: bool = False,
    * This specifies that serialization takes the compressed form of input ciphertexts.

You can see the impact of compression by comparing the size of the serialized form of input ciphertexts and evaluation keys with a sample code:

```python
from concrete import fhe
def test_compression(compression):
    @fhe.compiler({"counter": "encrypted"})
    def f(counter):
       return counter // 2

    circuit = f.compile(fhe.inputset(fhe.tensor[fhe.uint2, 3]),
                        compress_evaluation_keys=compression,
                        compress_input_ciphertexts=compression)

    print(f"Sizes with compression = {compression}")
    print(f" - of the input ciphertext {len(circuit.encrypt(list([0 for i in range(3)])).serialize())}")
    print(f" - of the evaluation keys {len(circuit.keys.serialize())}")

test_compression(False)
test_compression(True)
```
## Compression algorithms

The compression factor largely depends on the cryptographic parameters identified and the compression algorithms selected during the compilation.

Currently, **Concrete** uses the seeded compression algorithms. These algorithms rely on the fact that CSPRNGs are deterministic. Consequently, the chain of random values can be replaced by the seed and later recalculated using the same seed.

Typically, the size of a ciphertext is `(lwe dimension + 1) * 8` bytes, while the size of a seeded ciphertext is constant, equal to `3 * 8` bytes. Thus, the compression factor ranges from a hundred to thousands. Understanding the compression factor of evaluation keys is complex. The compression factor of evaluation keys typically ranges between 0 and 10.

{% hint style="warning" %}

Please note that while compression may save bandwidth and disk space, it incurs the cost of decompression. Currently, decompression occur more or less lazily during FHE evaluation without any control.

{% endhint %}
