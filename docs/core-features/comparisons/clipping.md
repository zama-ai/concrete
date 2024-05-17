# Clipping

This document explains the clipping strategy for comparison operations and demonstrates two implementation methods.

## Introduction

The clipping strategy optimizes the required bit width of the subtraction strategy - it clips the bigger operand and then does the subtraction with fewer bits.

For example, the results of `compare(3, 40)` and `compare(3, 4)` are the same,  so clipping the bigger operand before subtraction reduces the bit width needed.

### Requirements

The clipping method clips the bigger operand and then does the subtraction with fewer bits:

1. The clipping requires the operands to have **different** bit widths.
2. The subtraction then requires the operands to have the **same** bit widths.

For example,  when comparing `uint3` and `uint6`, we need to convert both of them to `uint4` in some way to do the subtraction and proceed with the Table Lookup (TLU) in 4 bits. In other words:

```python
x.bit_width != y.bit_width
```

```python
smaller = x if x.bit_width < y.bit_width else y
bigger = x if x.bit_width > y.bit_width else y
clipped = lambda value: np.clip(value, smaller.min() - 1, smaller.max() + 1)
any(
    (
        bit_width <= MAXIMUM_TLU_BIT_WIDTH and
        bit_width <= bigger.dtype.bit_width and
        bit_width > smaller.dtype.bit_width
    )
    for bit_width in [
        (smaller - clipped(bigger)).bit_width,
        (clipped(bigger) - smaller).bit_width,
    ]
  )
```

There are 2 methods to implement clipping.

## 1. Three TLUs: bigger clipped and smaller cast

### How it works

This method does not constrain bit widths during bit width assignment. It casts the smaller operand to a bit width that can store `clipped(bigger) - smaller` or `smaller - clipped(bigger)` during runtime using TLUs. It works like this:

```python
uint3_to_uint4_lut = fhe.LookupTable([...])
x_cast_to_uint4 = uint3_to_uint4_lut[x]

clipper = fhe.LookupTable([...])
y_clipped = clipper[y]

comparison_lut = fhe.LookupTable([...])
result = comparison_lut[x_cast_to_uint4 - y_clipped]
# or
another_comparison_lut = fhe.LookupTable([...])
result = another_comparison_lut[y_clipped - x_cast_to_uint4]
```

### How to use in FHE

The following example demonstrates how to implement clipping with three TLUs (bigger clipped and smaller cast) using **Concrete** to compare two encrypted integers:

```python
import numpy as np
from concrete import fhe

configuration = fhe.Configuration(
    comparison_strategy_preference=fhe.ComparisonStrategy.THREE_TLU_BIGGER_CLIPPED_SMALLER_CASTED
)

def f(x, y):
    return x < y

inputset = [
    (np.random.randint(0, 2**3), np.random.randint(0, 2**6))
    for _ in range(100)
]

compiler = fhe.Compiler(f, {"x": "encrypted", "y": "encrypted"})
circuit = compiler.compile(inputset, configuration, show_mlir=True)
```

<details>

<summary>The MLIR code generated</summary>

```cpp
module {
  
  // no promotions
  func.func @main(%arg0: !FHE.eint<3>, %arg1: !FHE.eint<6>) -> !FHE.eint<1> {
    
    // casting the smaller operand 
    %cst = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi64>
    %0 = "FHE.apply_lookup_table"(%arg0, %cst) : (!FHE.eint<3>, tensor<8xi64>) -> !FHE.esint<4>
    
    // clipping the bigger operand
    %cst_0 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]> : tensor<64xi64>
    %1 = "FHE.apply_lookup_table"(%arg1, %cst_0) : (!FHE.eint<6>, tensor<64xi64>) -> !FHE.esint<4>
    
    // subtraction
    %2 = "FHE.sub_eint"(%0, %1) : (!FHE.esint<4>, !FHE.esint<4>) -> !FHE.esint<4>
    
    // computing the result
    %cst_1 = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]> : tensor<16xi64>
    %3 = "FHE.apply_lookup_table"(%2, %cst_1) : (!FHE.esint<4>, tensor<16xi64>) -> !FHE.eint<1>
    
    return %3 : !FHE.eint<1>
    
  }
  
}
```

</details>

### Considerations

* **When to use:** when the difference of operands bit width is 1 bit (or in some cases 2 bits), and the subtraction is not optimal, you can use this clipping method as a fallback implementation instead of `fhe.ComparisonStrategy.CHUNKED`.
* **Number of TLU:** if the smaller operand is assigned a bit width bigger than or equal to the bit width that can store `clipped(bigger) - smaller` or `smaller - clipped(bigger)`, this method results in two TLUs.
* **Pros and cons:**
  * **Pros:**
    * **No bit width constraints:** good when the operands are used in other costly operations
    * **Efficient lookups:** requiring at most three TLUs
    * **Performance:** smaller bit widths used in TLUs lead to better performance.
  * **Cons:**
    * **Comparing integers of the same bit width:** not applicable

## 2. Two TLUs: bigger clipped and smaller promoted

### How it works

This method constrains the smaller operand to have at least the required bit width to store `clipped(bigger) - smaller` or `smaller - clipped(bigger)`, and clips the bigger operand to that bit width during runtime. It works like this:

```python
clipper = fhe.LookupTable([...])
y_clipped = clipper[y]

comparison_lut = fhe.LookupTable([...])
result = comparison_lut[x_promoted_to_uint4 - y_clipped]
# or
another_comparison_lut = fhe.LookupTable([...])
result = another_comparison_lut[y_clipped - x_promoted_to_uint4]
```

#### How to use in FHE

The following example demonstrates how to implement clipping with two TLUs (bigger clipped and smaller promoted) using **Concrete** to compare two encrypted integers:

```python
import numpy as np
from concrete import fhe

configuration = fhe.Configuration(
    comparison_strategy_preference=fhe.ComparisonStrategy.TWO_TLU_BIGGER_CLIPPED_SMALLER_PROMOTED
)

def f(x, y):
    return x < y

inputset = [
    (np.random.randint(0, 2**3), np.random.randint(0, 2**6))
    for _ in range(100)
]

compiler = fhe.Compiler(f, {"x": "encrypted", "y": "encrypted"})
circuit = compiler.compile(inputset, configuration, show_mlir=True)
```

<details>

<summary>The MLIR code generated</summary>

```cpp
module {
  
  // promotions          ............
  func.func @main(%arg0: !FHE.eint<4>, %arg1: !FHE.eint<6>) -> !FHE.eint<1> {
    
    // clipping the bigger operand
    %cst = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]> : tensor<64xi64>
    %0 = "FHE.apply_lookup_table"(%arg1, %cst) : (!FHE.eint<6>, tensor<64xi64>) -> !FHE.esint<4>
    
    // subtraction
    %1 = "FHE.to_signed"(%arg0) : (!FHE.eint<4>) -> !FHE.esint<4>
    %2 = "FHE.sub_eint"(%1, %0) : (!FHE.esint<4>, !FHE.esint<4>) -> !FHE.esint<4>
        
    // computing the result
    %cst_0 = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]> : tensor<16xi64>
    %3 = "FHE.apply_lookup_table"(%2, %cst_0) : (!FHE.esint<4>, tensor<16xi64>) -> !FHE.eint<1>
    
    return %3 : !FHE.eint<1>
    
  }
  
}
```

</details>

### Considerations

**Pros:**

* **Minimal constraints**: only the smaller operand is constrained, which is beneficial if the bigger operand is used in other costly operations.
* **Efficient TLU**: requires exactly two TLUs.

#### Cons

* **Potential slowdowns:** this method increases the bit width of both operands and locks them together across the whole circuit, potentially causing slowdowns in other costly operations.
