# Shift operations

This document explains how to configure and implement encrypted shift operations using promotion or casting methods in **Concrete.**

Although shifts are complex to implement, the key point is that the final result is computed using additions or subtractions on the original shifted operand. Since additions and subtractions require operands of the same bit width, the input and output bit widths must be synchronized.

There are two ways to perform shift operations.

### With promotion

The promotion method assigns the same bit width to the shifted operand and the shift result, which avoids an additional TLU on the shifted operand. However, it might increase the bit width of the result or the shifted operand. If they're used in other costly operations, it could cause significant slowdowns.

This method is the default one. Here is how to use it:

```python
import numpy as np
from concrete import fhe

configuration = fhe.Configuration(
    shifts_with_promotion=True,
)

def f(x, y):
    return x << y

inputset = [
    (np.random.randint(0, 2**3), np.random.randint(0, 2**2))
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
  func.func @main(%arg0: !FHE.eint<6>, %arg1: !FHE.eint<2>) -> !FHE.eint<6> {
    
    // shifting for the second bit of y
    %cst = arith.constant dense<[0, 0, 1, 1]> : tensor<4xi64>
    %0 = "FHE.apply_lookup_table"(%arg1, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<4>
    %cst_0 = arith.constant dense<[0, 0, 0, 2, 2, 2, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<64xi64>
    %1 = "FHE.apply_lookup_table"(%arg0, %cst_0) : (!FHE.eint<6>, tensor<64xi64>) -> !FHE.eint<4>
    %2 = "FHE.add_eint"(%1, %0) : (!FHE.eint<4>, !FHE.eint<4>) -> !FHE.eint<4>
    %cst_1 = arith.constant dense<[0, 0, 0, 8, 0, 16, 0, 24, 0, 32, 0, 40, 0, 48, 0, 56]> : tensor<16xi64>
    %3 = "FHE.apply_lookup_table"(%2, %cst_1) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.eint<6>
    %cst_2 = arith.constant dense<[0, 6, 12, 2, 8, 14, 4, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<64xi64>
    %4 = "FHE.apply_lookup_table"(%arg0, %cst_2) : (!FHE.eint<6>, tensor<64xi64>) -> !FHE.eint<4>
    %5 = "FHE.add_eint"(%4, %0) : (!FHE.eint<4>, !FHE.eint<4>) -> !FHE.eint<4>
    %cst_3 = arith.constant dense<[0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7]> : tensor<16xi64>
    %6 = "FHE.apply_lookup_table"(%5, %cst_3) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.eint<6>
    %7 = "FHE.add_eint"(%3, %6) : (!FHE.eint<6>, !FHE.eint<6>) -> !FHE.eint<6>
    %8 = "FHE.add_eint"(%7, %arg0) : (!FHE.eint<6>, !FHE.eint<6>) -> !FHE.eint<6>
    
    // shifting for the first bit of y
    %cst_4 = arith.constant dense<[0, 1, 0, 1]> : tensor<4xi64>
    %9 = "FHE.apply_lookup_table"(%arg1, %cst_4) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<4>
    %cst_5 = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8, 10, 10, 10, 10, 10, 10, 10, 10, 12, 12, 12, 12, 12, 12, 12, 12, 14, 14, 14, 14, 14, 14, 14, 14]> : tensor<64xi64>
    %10 = "FHE.apply_lookup_table"(%8, %cst_5) : (!FHE.eint<6>, tensor<64xi64>) -> !FHE.eint<4>
    %11 = "FHE.add_eint"(%10, %9) : (!FHE.eint<4>, !FHE.eint<4>) -> !FHE.eint<4>
    %12 = "FHE.apply_lookup_table"(%11, %cst_1) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.eint<6>
    %cst_6 = arith.constant dense<[0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14]> : tensor<64xi64>
    %13 = "FHE.apply_lookup_table"(%8, %cst_6) : (!FHE.eint<6>, tensor<64xi64>) -> !FHE.eint<4>
    %14 = "FHE.add_eint"(%13, %9) : (!FHE.eint<4>, !FHE.eint<4>) -> !FHE.eint<4>
    %15 = "FHE.apply_lookup_table"(%14, %cst_3) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.eint<6>
    %16 = "FHE.add_eint"(%12, %15) : (!FHE.eint<6>, !FHE.eint<6>) -> !FHE.eint<6>
    %17 = "FHE.add_eint"(%16, %8) : (!FHE.eint<6>, !FHE.eint<6>) -> !FHE.eint<6>
        
    return %17 : !FHE.eint<6>
        
  }
  
}
```

</details>

### With casting

For some circuits,  the promotion method is too complex to be optimal. When the promotion is disabled, you can perform shift operations this way:

```python
import numpy as np
from concrete import fhe

configuration = fhe.Configuration(
    shifts_with_promotion=False,
)

def f(x, y):
    return x << y

inputset = [
    (np.random.randint(0, 2**3), np.random.randint(0, 2**2))
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
  func.func @main(%arg0: !FHE.eint<3>, %arg1: !FHE.eint<2>) -> !FHE.eint<6> {
    
    // shifting for the second bit of y
    %cst = arith.constant dense<[0, 0, 1, 1]> : tensor<4xi64>
    %0 = "FHE.apply_lookup_table"(%arg1, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<4>
    %cst_0 = arith.constant dense<[0, 0, 0, 2, 2, 2, 4, 4]> : tensor<8xi64>
    %1 = "FHE.apply_lookup_table"(%arg0, %cst_0) : (!FHE.eint<3>, tensor<8xi64>) -> !FHE.eint<4>
    %2 = "FHE.add_eint"(%1, %0) : (!FHE.eint<4>, !FHE.eint<4>) -> !FHE.eint<4>
    %cst_1 = arith.constant dense<[0, 0, 0, 8, 0, 16, 0, 24, 0, 32, 0, 40, 0, 48, 0, 56]> : tensor<16xi64>
    %3 = "FHE.apply_lookup_table"(%2, %cst_1) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.eint<6>
    %cst_2 = arith.constant dense<[0, 6, 12, 2, 8, 14, 4, 10]> : tensor<8xi64>
    %4 = "FHE.apply_lookup_table"(%arg0, %cst_2) : (!FHE.eint<3>, tensor<8xi64>) -> !FHE.eint<4>
    %5 = "FHE.add_eint"(%4, %0) : (!FHE.eint<4>, !FHE.eint<4>) -> !FHE.eint<4>
    %cst_3 = arith.constant dense<[0, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7]> : tensor<16xi64>
    %6 = "FHE.apply_lookup_table"(%5, %cst_3) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.eint<6>
    %7 = "FHE.add_eint"(%3, %6) : (!FHE.eint<6>, !FHE.eint<6>) -> !FHE.eint<6>
        
    // cast x to 6-bits to compute the result using addition/subtraction
    %cst_4 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi64>
    %8 = "FHE.apply_lookup_table"(%arg0, %cst_4) : (!FHE.eint<3>, tensor<8xi64>) -> !FHE.eint<6>
    // this was done using promotion instead of casting in runtime when the flag was turned on
        
    %9 = "FHE.add_eint"(%7, %8) : (!FHE.eint<6>, !FHE.eint<6>) -> !FHE.eint<6>
        
    // shifting for the first bit of y
    %cst_5 = arith.constant dense<[0, 1, 0, 1]> : tensor<4xi64>
    %10 = "FHE.apply_lookup_table"(%arg1, %cst_5) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<4>
    %cst_6 = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8, 10, 10, 10, 10, 10, 10, 10, 10, 12, 12, 12, 12, 12, 12, 12, 12, 14, 14, 14, 14, 14, 14, 14, 14]> : tensor<64xi64>
    %11 = "FHE.apply_lookup_table"(%9, %cst_6) : (!FHE.eint<6>, tensor<64xi64>) -> !FHE.eint<4>
    %12 = "FHE.add_eint"(%11, %10) : (!FHE.eint<4>, !FHE.eint<4>) -> !FHE.eint<4>
    %13 = "FHE.apply_lookup_table"(%12, %cst_1) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.eint<6>
    %cst_7 = arith.constant dense<[0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14]> : tensor<64xi64>
    %14 = "FHE.apply_lookup_table"(%9, %cst_7) : (!FHE.eint<6>, tensor<64xi64>) -> !FHE.eint<4>
    %15 = "FHE.add_eint"(%14, %10) : (!FHE.eint<4>, !FHE.eint<4>) -> !FHE.eint<4>
    %16 = "FHE.apply_lookup_table"(%15, %cst_3) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.eint<6>
    %17 = "FHE.add_eint"(%13, %16) : (!FHE.eint<6>, !FHE.eint<6>) -> !FHE.eint<6>
    %18 = "FHE.add_eint"(%17, %9) : (!FHE.eint<6>, !FHE.eint<6>) -> !FHE.eint<6>
        
    return %18 : !FHE.eint<6>
  }
  
}
```

</details>
