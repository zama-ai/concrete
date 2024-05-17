# Table Lookups (TLUs)

This document outlines one of the most important operations in **Concrete -** Table Lookup (TLU).

## Introduction

**Concrete** converts most operations into TLUs under the hood, except for:

* Addition
* Subtraction
* Multiplication with non-encrypted values
* Tensor manipulation
* Other operations built from these primitives, such as matmul, conv. and so on.

TLUs are very flexible, enabling **Concrete** to support a wide range of operations. Even though the exact cost of TLUs depends on many variables such as hardware used and error probabilities, TLUs are always much more expensive than other operations.

Therefore, when feasible, you should reduce the number of TLUs or replace some of them with other primitive operations.

{% hint style="info" %}
**Concrete** automatically parallelizes TLUs when applied to tensors.
{% endhint %}

## How to perform TLUs

You can create and apply TLUs directly, refer to

* [Applying Table Lookups](applying-table-lookups.md)

Concrete provides several features to optimize the performance of TLUs, see

* [Bit extraction](bit\_extraction.md)
* [Rounding ](rounding.md)
* [Truncating](truncating.md)

## Exactness

TLUs are performed with a Fully Homomorphic Encryption (FHE) operation called [Programmable Bootstrapping](../fhe\_basics.md#function-evaluation) (PBS). PBSs have a certain probability of error. When these errors occur, they may cause inaccurate results.

Consider the following table:

```python
lut = [0, 1, 4, 9, 16, 25, 36, 49, 64]
```

If you perform a TLU using `4`, the expected result is `lut[4] = 16`. However, due to the possible error, it can return any other value in the table.

### Configurations

You can configure the probability error using the `p_error` and `global_p_error` options.

* `p_error` applies to individual TLUs
* `global_p_error` applies to the whole circuit

For example, if you set `p_error` to `0.01`, each TLU in the circuit will have a 99% (or greater) chance of being exact. It means:

* With only one TLU in the circuit, it corresponds to `global_p_error = 0.01`.
* With two TLUs, `global_p_error` would be higher: `1 - (0.99 * 0.99) ≈ 0.02 = 2%`.

Setting `global_p_error` to `0.01` ensures that the entire circuit will have at most a `1%` probability of error, regardless of the number of TLUs. In this case, `p_error` will be smaller than `0.01` if there is more than one TLU.

By default, both `p_error` and `global_p_error` are set to `None`, which implies that the `global_p_error` is `1 / 100_000`. If both `p_error` and `global_p_error` are set, the stricter condition will apply.

See [How to Configure](../../guides/configure.md) to learn how you can set a custom `p_error` and/or `global_p_error`.

{% hint style="info" %}
Configuring these variables has impacts on:

* **Compilation and execution times**: compilation, key generation, circuit execution
* **Space requirements**: key sizes on disk and in memory.

In general, lower error probabilities result in longer compilation and execution times and larger space requirements.
{% endhint %}

## Performance

PBSs are computationally expensive. In some cases, you can replace PBS with [rounded PBS](rounding.md), [truncate PBS](truncating.md), or [approximate PBS](rounding.md) to optimize the performance. With slightly different semantics, these TLUs offer more efficiency without sacrificing accuracy, which can be very useful in cases like machine learning.
