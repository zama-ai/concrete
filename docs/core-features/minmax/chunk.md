# Chunk

This document explains the chunk strategy for min/max operations on encrypted data.

The chunk strategy breaks down the operands into smaller chunks and then processes chunks using Table Lookups (TLUs) with multiplications and additions. This general strategy is suitable for any situation.

## How it works

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

## How to use in FHE

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

<details>

<summary>The MLIR generated</summary>

```cpp
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

</details>

## Considerations

* **Computational cost:**
  * The initial comparison is [chunked](../comparisons/#chunked) as well, which is already very expensive.
  * Multiplication with operands can not increase the bit width of the inputs, adding more computational cost.
* **Chunk size: Concrete** automatically selects the optimal chunk size to reduce the number of TLUs.
* **Number of TLUs:** between 9 to 21.
* **When to use:** when no other implementation can be used
* **Pros and cons:**
  * **Pros:** compatible with any integers
  * **Cons:** extremely expensive
