# Min/Max Operations

This document explains how to compute minimum and maximum between values in Concrete, covering different strategies to make computations faster, depending on the strategy.

Finding the minimum or maximum of two numbers is not a native operation in Concrete, so it needs to be implemented using existing native operations (i.e., additions, clear multiplications, negations, table lookups). Concrete offers two different implementations for this.

## Chunked

This is the most general implementation that can be used in any situation. The idea is:

```python
# (example below is for bit-width of 8 and chunk size of 4)

# compare lhs and rhs
select_lhs = lhs < rhs  # or lhs > rhs for maximum

# multiply lhs with select_lhs
lhs_contribution = lhs * select_lhs

# multiply rhs with 1 - select_lhs
rhs_contribution = rhs * (1 - select_lhs)

# compute the result
result = lhs_contribution + rhs_contribution
```

### Notes

- Initial comparison is [chunked](./comparisons.md#chunked) as well, which is already very expensive. 
- Multiplication with operands aren't allowed to increase the bit-width of the inputs, so they are very expensive as well.
- Optimal chunk size is selected automatically to reduce the number of table lookups.
- Chunked comparisons result in at least 9 and at most 21 table lookups.
- It is used if no other implementation can be used.

### Pros

- Can be used with any integers.

### Cons

- Extremely expensive.

### Example

```python
import numpy as np
from concrete import fhe

def f(x, y):
    return np.minimum(x, y)

inputset = [
    (np.random.randint(0, 2**4), np.random.randint(0, 2**4))
    for _ in range(100)
]

compiler = fhe.Compiler(f, {"x": "encrypted", "y": "encrypted"})
circuit = compiler.compile(inputset, show_mlir=True)
```

produces

```c++
module {

  func.func @main(%arg0: !FHE.eint<4>, %arg1: !FHE.eint<4>) -> !FHE.eint<4> {
  
    // calculating select_x, which is x < y since we're computing the minimum
    %cst = arith.constant dense<[0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12]> : tensor<16xi64>
    %0 = "FHE.apply_lookup_table"(%arg0, %cst) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.eint<4>
    %cst_0 = arith.constant dense<[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]> : tensor<16xi64>
    %1 = "FHE.apply_lookup_table"(%arg1, %cst_0) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.eint<4>
    %2 = "FHE.add_eint"(%0, %1) : (!FHE.eint<4>, !FHE.eint<4>) -> !FHE.eint<4>
    %cst_1 = arith.constant dense<[0, 1, 1, 1, 2, 0, 1, 1, 2, 2, 0, 1, 2, 2, 2, 0]> : tensor<16xi64>
    %3 = "FHE.apply_lookup_table"(%2, %cst_1) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.eint<4>
    %cst_2 = arith.constant dense<[0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12]> : tensor<16xi64>
    %4 = "FHE.apply_lookup_table"(%arg0, %cst_2) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.eint<4>
    %cst_3 = arith.constant dense<[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]> : tensor<16xi64>
    %5 = "FHE.apply_lookup_table"(%arg1, %cst_3) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.eint<4>
    %6 = "FHE.add_eint"(%4, %5) : (!FHE.eint<4>, !FHE.eint<4>) -> !FHE.eint<4>
    %cst_4 = arith.constant dense<[0, 4, 4, 4, 8, 0, 4, 4, 8, 8, 0, 4, 8, 8, 8, 0]> : tensor<16xi64>
    %7 = "FHE.apply_lookup_table"(%6, %cst_4) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.eint<4>
    %8 = "FHE.add_eint"(%7, %3) : (!FHE.eint<4>, !FHE.eint<4>) -> !FHE.eint<4>
    %cst_5 = arith.constant dense<[0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]> : tensor<16xi64>
    %9 = "FHE.apply_lookup_table"(%8, %cst_5) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.eint<1>
    
    // extracting the first 2 bits of x shifhted to left by 1 bits for packing
    %cst_6 = arith.constant dense<[0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6]> : tensor<16xi64>
    %10 = "FHE.apply_lookup_table"(%arg0, %cst_6) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.eint<3>
    
    // casting select_x to 3 bits for packing
    %cst_7 = arith.constant dense<[0, 1]> : tensor<2xi64>
    %11 = "FHE.apply_lookup_table"(%9, %cst_7) : (!FHE.eint<1>, tensor<2xi64>) -> !FHE.eint<3>
    
    // packing the first 2 bits of x with select_x
    %12 = "FHE.add_eint"(%10, %11) : (!FHE.eint<3>, !FHE.eint<3>) -> !FHE.eint<3>
    
    // calculating contribution of 0 if select_x is 0 else the first 2 bits of x
    %cst_8 = arith.constant dense<[0, 0, 0, 1, 0, 2, 0, 3]> : tensor<8xi64>
    %13 = "FHE.apply_lookup_table"(%12, %cst_8) : (!FHE.eint<3>, tensor<8xi64>) -> !FHE.eint<4>
    
    // extracting the last 2 bits of x shifhted to the left by 1 bit for packing
    %cst_9 = arith.constant dense<[0, 0, 0, 0, 2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6]> : tensor<16xi64>
    %14 = "FHE.apply_lookup_table"(%arg0, %cst_9) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.eint<3>
    
    // packing the last 2 bits of x with select_x
    %15 = "FHE.add_eint"(%14, %11) : (!FHE.eint<3>, !FHE.eint<3>) -> !FHE.eint<3>
    
    // calculating contribution of 0 if select_x is 0 else the last 2 bits of x shifted by 2 bits for direct addition
    %cst_10 = arith.constant dense<[0, 0, 0, 4, 0, 8, 0, 12]> : tensor<8xi64>
    %16 = "FHE.apply_lookup_table"(%15, %cst_10) : (!FHE.eint<3>, tensor<8xi64>) -> !FHE.eint<4>
    
    // computing x * select_x
    %17 = "FHE.add_eint"(%13, %16) : (!FHE.eint<4>, !FHE.eint<4>) -> !FHE.eint<4>
    
    // extracting the first 2 bits of y shifhted to the left by 1 bit for packing
    %18 = "FHE.apply_lookup_table"(%arg1, %cst_6) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.eint<3>
    
    // packing the first 2 bits of y with select_x
    %19 = "FHE.add_eint"(%18, %11) : (!FHE.eint<3>, !FHE.eint<3>) -> !FHE.eint<3>
    
    // calculating contribution of 0 if select_x is 1 else the first 2 bits of y
    %cst_11 = arith.constant dense<[0, 0, 1, 0, 2, 0, 3, 0]> : tensor<8xi64>
    %20 = "FHE.apply_lookup_table"(%19, %cst_11) : (!FHE.eint<3>, tensor<8xi64>) -> !FHE.eint<4>
    
    // extracting the last 2 bits of y shifhted to left by 1 bit for packing
    %21 = "FHE.apply_lookup_table"(%arg1, %cst_9) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.eint<3>
    
    // packing the last 2 bits of y with select_x
    %22 = "FHE.add_eint"(%21, %11) : (!FHE.eint<3>, !FHE.eint<3>) -> !FHE.eint<3>
    
    // calculating contribution of 0 if select_x is 1 else the last 2 bits of y shifted by 2 bits for direct addition
    %cst_12 = arith.constant dense<[0, 0, 4, 0, 8, 0, 12, 0]> : tensor<8xi64>
    %23 = "FHE.apply_lookup_table"(%22, %cst_12) : (!FHE.eint<3>, tensor<8xi64>) -> !FHE.eint<4>
    
    // computing y * (1 - select_x)
    %24 = "FHE.add_eint"(%20, %23) : (!FHE.eint<4>, !FHE.eint<4>) -> !FHE.eint<4>
    
    // computing the result
    %25 = "FHE.add_eint"(%17, %24) : (!FHE.eint<4>, !FHE.eint<4>) -> !FHE.eint<4>

    return %25 : !FHE.eint<4>
    
  }
  
}
```

## Min/Max Trick

This implementation uses the fact that `[min,max](x, y)` is equal to `[min, max](x - y, 0) + y`, which is just a subtraction, a table lookup and an addition!

There are two major problems with this implementation though:
1) subtraction before the TLU requires up to 2 additional bits to avoid overflows (it is 1 in most cases).
2) subtraction and addition require the same bit-width across operands.

What this means is that if we are comparing `uint3` and `uint6`, we need to convert both of them to `uint7` in some way to do the subtraction and proceed with the TLU in 7-bits. There are 2 ways to achieve this behavior.

### Requirements

- ```python
  (x - y).bit_width <= MAXIMUM_TLU_BIT_WIDTH
  ```

### 1. fhe.MinMaxStrategy.ONE_TLU_PROMOTED

This strategy makes sure that during bit-width assignment, both operands are assigned the same bit-width, and that bit-width contains at least the amount of bits required to store `x - y`. The idea is:

```python
comparison_lut = fhe.LookupTable([...])
result = comparison_lut[x_promoted_to_uint7 - y_promoted_to_uint7] + y_promoted_to_uint7
```

#### Pros

- It will always result in a single table lookup.

#### Cons

- It will increase the bit-width of both operands and the result, and lock them together across the whole circuit, which can result in significant slowdowns if the result or the operands are used in other costly operations.

#### Example

```python
import numpy as np
from concrete import fhe

configuration = fhe.Configuration(
    min_max_strategy_preference=fhe.MinMaxStrategy.ONE_TLU_PROMOTED,
)

def f(x, y):
    return np.minimum(x, y)

inputset = [
    (np.random.randint(0, 2**4), np.random.randint(0, 2**2))
    for _ in range(100)
]

compiler = fhe.Compiler(f, {"x": "encrypted", "y": "encrypted"})
circuit = compiler.compile(inputset, configuration, show_mlir=True)
```

produces

```c++
module {

  // promotions          ............         ............
  func.func @main(%arg0: !FHE.eint<5>, %arg1: !FHE.eint<5>) -> !FHE.eint<5> {
  
    // subtraction
    %0 = "FHE.to_signed"(%arg0) : (!FHE.eint<5>) -> !FHE.esint<5>
    %1 = "FHE.to_signed"(%arg1) : (!FHE.eint<5>) -> !FHE.esint<5>
    %2 = "FHE.sub_eint"(%0, %1) : (!FHE.esint<5>, !FHE.esint<5>) -> !FHE.esint<5>
    
    // tlu
    %cst = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1]> : tensor<32xi64>
    %3 = "FHE.apply_lookup_table"(%2, %cst) : (!FHE.esint<5>, tensor<32xi64>) -> !FHE.eint<5>
    
    // addition
    %4 = "FHE.add_eint"(%3, %arg1) : (!FHE.eint<5>, !FHE.eint<5>) -> !FHE.eint<5>
    
    return %4 : !FHE.eint<5>
    
  }
  
}
```

### 2. fhe.MinMaxStrategy.THREE_TLU_CASTED

This strategy will not put any constraint on bit-widths during bit-width assignment. Instead, operands are cast to a bit-width that can store `x - y` during runtime using table lookups. The idea is:

```python
uint3_to_uint7_lut = fhe.LookupTable([...])
x_cast_to_uint7 = uint3_to_uint7_lut[x]

uint6_to_uint7_lut = fhe.LookupTable([...])
y_cast_to_uint7 = uint6_to_uint7_lut[y]

comparison_lut = fhe.LookupTable([...])
result = comparison_lut[x_cast_to_uint7 - y_cast_to_uint7] + y
```

#### Notes

- It can result in a single table lookup as well, if x and y are assigned (because of other operations) the same bit-width, and that bit-width can store `x - y`.
- Or in two table lookups if only one of the operands is assigned a bit-width bigger than or equal to the bit width that can store `x - y`.

#### Pros

- It will not put any constraints on bit-widths of the operands, which is amazing if they are used in other costly operations.
- It will result in at most 3 table lookups, which is still good.

#### Cons

- If you are not doing anything else with the operands, or doing less costly operations compared to comparison, it will introduce up to two unnecessary table lookups and slow down execution compared to `fhe.MinMaxStrategy.ONE_TLU_PROMOTED`.

#### Example

```python
import numpy as np
from concrete import fhe

configuration = fhe.Configuration(
    min_max_strategy_preference=fhe.MinMaxStrategy.THREE_TLU_CASTED,
)

def f(x, y):
    return np.minimum(x, y)

inputset = [
    (np.random.randint(0, 2**4), np.random.randint(0, 2**2))
    for _ in range(100)
]

compiler = fhe.Compiler(f, {"x": "encrypted", "y": "encrypted"})
circuit = compiler.compile(inputset, configuration, show_mlir=True)
```

produces

```c++
module {

  // no promotions
  func.func @main(%arg0: !FHE.eint<4>, %arg1: !FHE.eint<2>) -> !FHE.eint<2> {
  
    // casting x
    %cst = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : tensor<16xi64>
    %0 = "FHE.apply_lookup_table"(%arg0, %cst) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.esint<5>
    
    // casting y
    %cst_0 = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi64>
    %1 = "FHE.apply_lookup_table"(%arg1, %cst_0) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.esint<5>
    
    // subtraction
    %2 = "FHE.sub_eint"(%0, %1) : (!FHE.esint<5>, !FHE.esint<5>) -> !FHE.esint<5>
    
    // tlu
    %cst_1 = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1]> : tensor<32xi64>
    %3 = "FHE.apply_lookup_table"(%2, %cst_1) : (!FHE.esint<5>, tensor<32xi64>) -> !FHE.eint<2>
    
    // addition
    %4 = "FHE.add_eint"(%3, %arg1) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    
    return %4 : !FHE.eint<2>
    
  }
  
}
```

## Summary

| Strategy                                | Minimum # of TLUs | Maximum # of TLUs | Can increase the bit-width of the inputs |
|-----------------------------------------|-------------------|-------------------|------------------------------------------|
| CHUNKED                                 | 9                 | 21                |                                          |
| ONE_TLU_PROMOTED                        | 1                 | 1                 | ✓                                        |
| THREE_TLU_CASTED                        | 1                 | 3                 |                                          |

{% hint style="info" %}
Concrete will choose the best strategy available after bit-width assignment, regardless of the specified preference.
{% endhint %}

{% hint style="info" %}
Different strategies are good for different circuits. If you want the best runtime for your use case, you can compile your circuit with all different comparison strategy preferences, and pick the one with the lowest complexity.
{% endhint %}
