# Bitwise operations

This document outlines two strategies to perform bitwise operations over encrypted data in **Concrete.**

## Introduction

**Concrete** doesn't natively support bitwise operations. To perform bitwise operations, you must use native operations like additions, clear multiplications, negations, and table lookups (TLUs).

There are 2 different methods:

* [Chunk](chunk.md): the chunk strategy breaks down the operands into smaller chunks and then processes chunks using TLUs with multiplications and additions. This general strategy is suitable for any situation.
* [Packing](packing.md): the packing strategy combines two values into a single value and applies a single TLU to this combined value.

Shift operations are more complex to implement. Refer to[ shift operations](shift_operations.md) for details explanations.

{% hint style="info" %}
**Implementation tips:**

* **Concrete** automatically selects the optimal strategy after bit width assignment, regardless of the specified preference.
* Different strategies work best for different circuits. To achieve the best runtime for your use case, compile your circuit using each comparison strategy preference and choose the one with the lowest complexity.
{% endhint %}

## Summary

| Strategy                                    | Minimum # of TLUs | Maximum # of TLUs | Can increase the bit-width of the inputs |
| ------------------------------------------- | ----------------- | ----------------- | ---------------------------------------- |
| CHUNKED                                     | 4                 | 9                 |                                          |
| ONE\_TLU\_PROMOTED                          | 1                 | 1                 | ✓                                        |
| THREE\_TLU\_CASTED                          | 1                 | 3                 |                                          |
| TWO\_TLU\_BIGGER\_PROMOTED\_SMALLER\_CASTED | 1                 | 2                 | ✓                                        |
| TWO\_TLU\_BIGGER\_CASTED\_SMALLER\_PROMOTED | 1                 | 2                 | ✓                                        |
