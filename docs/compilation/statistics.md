# Statistics
This document provides an overview of how to analyze compiled circuits and extract statistical data for performance evaluation in **Concrete**. These statistics help identify bottlenecks and compare circuits.

## Basic operations
Concrete calculates statistics based on the following six basic operations:

* **Clear addition:** `x + y` where `x` is encrypted and `y` is clear
* **Encrypted addition:** `x + y` where both `x` and `y` are encrypted
* **Clear multiplication:** `x * y` where `x` is encrypted and `y` is clear
* **Encrypted negation:** `-x` where `x` is encrypted
* **Key switch:** A building block for table lookups
* **Packing key switch:** A building block for table lookups
* **Programmable bootstrapping:** A building block for table lookups

## Displaying statistics
You can print all statistics using the `show_statistics` configuration option:

```python
from concrete import fhe

@fhe.compiler({"x": "encrypted"})
def f(x):
    return (x**2) + (2*x) + 4

inputset = range(2**2)
circuit = f.compile(inputset, show_statistics=True)
```

This code will print:

```
Statistics
--------------------------------------------------------------------------------
size_of_secret_keys: 22648
size_of_bootstrap_keys: 51274176
size_of_keyswitch_keys: 64092720
size_of_inputs: 16392
size_of_outputs: 16392
p_error: 9.627450598589458e-06
global_p_error: 9.627450598589458e-06
complexity: 99198195.0
programmable_bootstrap_count: 1
programmable_bootstrap_count_per_parameter: {
    BootstrapKeyParam(polynomial_size=2048, glwe_dimension=1, input_lwe_dimension=781, level=1, base_log=23, variance=9.940977002694397e-32): 1
}
key_switch_count: 1
key_switch_count_per_parameter: {
    KeyswitchKeyParam(level=5, base_log=3, variance=1.939836732335308e-11): 1
}
packing_key_switch_count: 0
clear_addition_count: 1
clear_addition_count_per_parameter: {
    LweSecretKeyParam(dimension=2048): 1
}
encrypted_addition_count: 1
encrypted_addition_count_per_parameter: {
    LweSecretKeyParam(dimension=2048): 1
}
clear_multiplication_count: 1
clear_multiplication_count_per_parameter: {
    LweSecretKeyParam(dimension=2048): 1
}
encrypted_negation_count: 0
--------------------------------------------------------------------------------
```

{% hint style="info" %}
Each of these properties can be directly accessed on the circuit (e.g., `circuit.programmable_bootstrap_count`).
{% endhint %}

## Tags

You can also use tags to analyze specific sections of your circuit. See more detailed explanation in [tags documentation](../core-features/tagging.md).

Imagine you have a neural network with 10 layers, each of them tagged, you can easily see the number of additions and multiplications required for matrix multiplications per layer:

```
Statistics
--------------------------------------------------------------------------------
clear_multiplication_count_per_tag: {
    /model/model: 53342
    /model/model.0/Gemm: 14720
    /model/model.0/Gemm.matmul: 14720
    /model/model.2/Gemm: 11730
    /model/model.2/Gemm.matmul: 11730
    /model/model.4/Gemm: 9078
    /model/model.4/Gemm.matmul: 9078
    /model/model.6/Gemm: 6764
    /model/model.6/Gemm.matmul: 6764
    /model/model.8/Gemm: 4788
    /model/model.8/Gemm.matmul: 4788
    /model/model.10/Gemm: 3150
    /model/model.10/Gemm.matmul: 3150
    /model/model.12/Gemm: 1850
    /model/model.12/Gemm.matmul: 1850
    /model/model.14/Gemm: 888
    /model/model.14/Gemm.matmul: 888
    /model/model.16/Gemm: 264
    /model/model.16/Gemm.matmul: 264
    /model/model.18/Gemm: 110
    /model/model.18/Gemm.matmul: 110
}
encrypted_addition_count_per_tag: {
    /model/model: 53342
    /model/model.0/Gemm: 14720
    /model/model.0/Gemm.matmul: 14720
    /model/model.2/Gemm: 11730
    /model/model.2/Gemm.matmul: 11730
    /model/model.4/Gemm: 9078
    /model/model.4/Gemm.matmul: 9078
    /model/model.6/Gemm: 6764
    /model/model.6/Gemm.matmul: 6764
    /model/model.8/Gemm: 4788
    /model/model.8/Gemm.matmul: 4788
    /model/model.10/Gemm: 3150
    /model/model.10/Gemm.matmul: 3150
    /model/model.12/Gemm: 1850
    /model/model.12/Gemm.matmul: 1850
    /model/model.14/Gemm: 888
    /model/model.14/Gemm.matmul: 888
    /model/model.16/Gemm: 264
    /model/model.16/Gemm.matmul: 264
    /model/model.18/Gemm: 110
    /model/model.18/Gemm.matmul: 110
}
--------------------------------------------------------------------------------
```
