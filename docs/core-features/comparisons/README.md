# Comparisons

This document outlines three strategies to perform comparison operations over encrypted data in **Concrete.**

## Introduction

**Concrete** doesn't natively support comparison operations. To perform comparisons, you must use existing native operations like additions, clear multiplications, negations, and Table Lookups (TLUs).

There are 3 different strategies:

* [**Chunk:** ](chunk.md)The chunk strategy breaks down the operands into smaller chunks, which are then processed using TLUs. This general strategy is suitable for any situation.
* [**Subtraction**](subtraction.md)**:** The subtraction strategy converts `x [<,<=,==,!=,>=,>] y` to `x - y [<,<=,==,!=,>=,>] 0`, which is a simple subtraction and a TLU.
* [**Clipping**](clipping.md)**:** The clipping strategy optimizes the required bit width of the subtraction strategy - it clips the bigger operand and then does the subtraction with fewer bits.

{% hint style="info" %}
**Implementation tips:**

* **Concrete** automatically selects the optimal strategy after bit width assignment, regardless of the specified preference.
* Different strategies work best for different circuits. To achieve the best runtime for your use case, compile your circuit using each comparison strategy preference and choose the one with the lowest complexity.
{% endhint %}

## Summary

| Strategy                                     | Minimum # of TLUs | Maximum # of TLUs | Can increase the bit-width of the inputs |
| -------------------------------------------- | ----------------- | ----------------- | ---------------------------------------- |
| CHUNKED                                      | 5                 | 13                |                                          |
| ONE\_TLU\_PROMOTED                           | 1                 | 1                 | ✓                                        |
| THREE\_TLU\_CASTED                           | 1                 | 3                 |                                          |
| TWO\_TLU\_BIGGER\_PROMOTED\_SMALLER\_CASTED  | 1                 | 2                 | ✓                                        |
| TWO\_TLU\_BIGGER\_CASTED\_SMALLER\_PROMOTED  | 1                 | 2                 | ✓                                        |
| THREE\_TLU\_BIGGER\_CLIPPED\_SMALLER\_CASTED | 2                 | 3                 |                                          |
| TWO\_TLU\_BIGGER\_CLIPPED\_SMALLER\_PROMOTED | 2                 | 2                 | ✓                                        |
