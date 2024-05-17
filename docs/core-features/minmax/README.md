# Min/Max operations

This document outlines two strategies to perform min/max operations over encrypted data in **Concrete.**

## Introduction

**Concrete** doesn't natively support finding the minimum or maximum of two numbers. To perform min/max operations, you must use native operations like additions, clear multiplications, negations, and table lookups (TLUs).

There are 2 different strategies:

* [Chunk](chunk.md): the chunk strategy breaks down the operands into smaller chunks and then processes chunks using TLUs with multiplications and additions. This general strategy is suitable for any situation.
* [Subtraction](subtraction.md): the subtraction strategy converts `[min,max](x, y)` to `[min, max](x - y, 0) +`, which is a simple subtraction and a TLU.

{% hint style="info" %}
**Implementation tips:**

* **Concrete** automatically selects the optimal strategy after bit width assignment, regardless of the specified preference.
* Different strategies work best for different circuits. To achieve the best runtime for your use case, compile your circuit using each comparison strategy preference and choose the one with the lowest complexity.
{% endhint %}

## Summary

| Strategy           | Minimum # of TLUs | Maximum # of TLUs | Can increase the bit-width of the inputs |
| ------------------ | ----------------- | ----------------- | ---------------------------------------- |
| CHUNKED            | 9                 | 21                |                                          |
| ONE\_TLU\_PROMOTED | 1                 | 1                 | ✓                                        |
| THREE\_TLU\_CASTED | 1                 | 3                 |                                          |
