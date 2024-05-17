# Packing

This document explains the packing strategy for bitwise operations on encrypted data and demonstrates four implementation methods.

## Introduction

The packing strategy combines two values into a single value and applies a single Table Lookup (TLU) to this combined value.

## Requirements

The packing strategy requires that:

1. The operands  have the same bit width
2. The bid width is at least `x.bit_width + y.bit_width` and can't exceed the maximum TLU bit width, which is `16` at the moment.

For example,  when comparing `uint3` and `uint6`, we need to convert both of them to `uint9` in some way to do the subtraction and proceed with the TLU in 9 bits. In other words:

* ```python
  x.bit_width + y.bit_width <= MAXIMUM_TLU_BIT_WIDTH
  ```

There are 4 ways to implement this strategy.

## 1. One TLU promoted

### How it works

This method ensures that both operands are assigned the same bit width, which contains at least the number of bits required to store `pack(x, y)`. It works like this:

```python
bitwise_lut = fhe.LookupTable([...])
result = bitwise_lut[pack(x_promoted_to_uint9, y_promoted_to_uint9)]
```

### How to use in FHE

The following example demonstrates how to implement packing with one TLU promoted using **Concrete** to perform bitwise AND operations over two encrypted integers:

```python
import numpy as np
from concrete import fhe

configuration = fhe.Configuration(
    bitwise_strategy_preference=fhe.BitwiseStrategy.ONE_TLU_PROMOTED,
)

def f(x, y):
    return x & y

inputset = [
    (np.random.randint(0, 2**4), np.random.randint(0, 2**4))
    for _ in range(100)
]

compiler = fhe.Compiler(f, {"x": "encrypted", "y": "encrypted"})
circuit = compiler.compile(inputset, configuration, show_mlir=True)
```

<details>

<summary>The MLIR code generated</summary>

```cpp
module {
  
  // promotions          ............         ............
  func.func @main(%arg0: !FHE.eint<8>, %arg1: !FHE.eint<8>) -> !FHE.eint<4> {
    
    // packing
    %c16_i9 = arith.constant 16 : i9
    %0 = "FHE.mul_eint_int"(%arg0, %c16_i9) : (!FHE.eint<8>, i9) -> !FHE.eint<8>
    %1 = "FHE.add_eint"(%0, %arg1) : (!FHE.eint<8>, !FHE.eint<8>) -> !FHE.eint<8>
        
    // computing the result
    %cst = arith.constant dense<"..."> : tensor<256xi64>
    %2 = "FHE.apply_lookup_table"(%1, %cst) : (!FHE.eint<8>, tensor<256xi64>) -> !FHE.eint<4>
        
    return %2 : !FHE.eint<4>
        
  }
  
}
```

</details>

### Considerations

* **Pros:**
  * **Efficient TLU:**  results in only one single TLU
* **Cons:**
  * **Potential slowdowns:** this method increases the bit width of both operands and locks them together across the whole circuit, potentially causing slowdowns in other costly operations.

## 2. Three TLUs cast

### How it works

This method does not constrain bit widths during bit width assignment. It casts operands to a bit width that can store `pack(x, y)` during runtime using TLUs. It works like this:

```python
uint3_to_uint9_lut = fhe.LookupTable([...])
x_cast_to_uint9 = uint3_to_uint9_lut[x]

uint6_to_uint9_lut = fhe.LookupTable([...])
y_cast_to_uint9 = uint6_to_uint9_lut[y]

bitwise_lut = fhe.LookupTable([...])
result = bitwise_lut[pack(x_cast_to_uint9, y_cast_to_uint9)]
```

### How to use in FHE

The following example demonstrates how to implement subtraction with three TLUs cast using **Concrete** to perform bitwise AND operations over two encrypted integers:

```python
import numpy as np
from concrete import fhe

configuration = fhe.Configuration(
    comparison_strategy_preference=fhe.BitwiseStrategy.THREE_TLU_CASTED,
)

def f(x, y):
    return x & y

inputset = [
    (np.random.randint(0, 2**4), np.random.randint(0, 2**4))
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
  func.func @main(%arg0: !FHE.eint<4>, %arg1: !FHE.eint<4>) -> !FHE.eint<4> {
    
    // casting
    %cst = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : tensor<16xi64>
    %0 = "FHE.apply_lookup_table"(%arg0, %cst) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.eint<8>
    %1 = "FHE.apply_lookup_table"(%arg1, %cst) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.eint<8>

    // packing
    %c16_i9 = arith.constant 16 : i9
    %2 = "FHE.mul_eint_int"(%0, %c16_i9) : (!FHE.eint<8>, i9) -> !FHE.eint<8>
    %3 = "FHE.add_eint"(%2, %1) : (!FHE.eint<8>, !FHE.eint<8>) -> !FHE.eint<8>
        
    // computing the result
    %cst_0 = arith.constant dense<"..."> : tensor<256xi64>
    %4 = "FHE.apply_lookup_table"(%3, %cst_0) : (!FHE.eint<8>, tensor<256xi64>) -> !FHE.eint<4>
        
    return %4 : !FHE.eint<4>
        
  }
  
}
```

</details>

### Considerations

* **Number of TLUs:**
  * If `x` and `y` are assigned the same bit width which can store `pack(x, y)`because of other operations, this method uses only a single TLU.
  * If one of the operands is assigned a bit width bigger than or equal to the required bit width to store `pack(x, y),` this method can use 2 TLUs.
* **Pros and cons:**
  * **Pros:**
    * **No bit width constraints**: this method does not constraint the bit widths of the operands, which is beneficial when other costly operations are involved.
    * **Efficient TLU**: it requires at most three TLUs, which is still efficient.
  * **Cons:**
    * **Potential inefficiency:** if you don't use the operands for other operations, or do less costly operations than bitwise operations, this method introduces up to two unnecessary TLUs and slows down execution compared to `fhe.BitwiseStrategy.ONE_TLU_PROMOTED`.

## 3. Two TLUs: bigger promoted and smaller cast

### How it works

This method constrains only the bigger operand to have at least the bid width to store `pack(x, y)` and casts the smaller operand to that bid width during runtime. It works like this:

```python
uint3_to_uint9_lut = fhe.LookupTable([...])
x_cast_to_uint9 = uint3_to_uint9_lut[x]

comparison_lut = fhe.LookupTable([...])
result = comparison_lut[x_cast_to_uint9 - y_promoted_to_uint9]
```

### How to use in FHE

The following example demonstrates how to implement subtraction with 2 TLUs (bigger promoted and smaller cast) using **Concrete** to perform bitwise AND operations over two encrypted integers:

```python
import numpy as np
from concrete import fhe

configuration = fhe.Configuration(
    bitwise_strategy_preference=fhe.BitwiseStrategy.TWO_TLU_BIGGER_PROMOTED_SMALLER_CASTED,
)

def f(x, y):
    return x & y

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
  
  // promotions                               ............
  func.func @main(%arg0: !FHE.eint<3>, %arg1: !FHE.eint<8>) -> !FHE.eint<3> {
    
    // casting smaller operand
    %cst = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi64>
    %0 = "FHE.apply_lookup_table"(%arg0, %cst) : (!FHE.eint<3>, tensor<8xi64>) -> !FHE.eint<8>
        
    // packing
    %c32_i9 = arith.constant 32 : i9
    %1 = "FHE.mul_eint_int"(%0, %c32_i9) : (!FHE.eint<8>, i9) -> !FHE.eint<8>
    %2 = "FHE.add_eint"(%1, %arg1) : (!FHE.eint<8>, !FHE.eint<8>) -> !FHE.eint<8>
        
    // computing the result
    %cst_0 = arith.constant dense<"..."> : tensor<256xi64>
    %3 = "FHE.apply_lookup_table"(%2, %cst_0) : (!FHE.eint<8>, tensor<256xi64>) -> !FHE.eint<3>
        
    return %3 : !FHE.eint<3>
        
  }
  
}
```

</details>

### Considerations

* **The number of TLUs:** if the smaller operand is assigned the same bit width as the bigger operand because of other operations, this method uses only a single TLU.
* **Pros and cons:**
  * **Pros:**
    * **Minimal constraints**: only the bigger operand is constrained, which is beneficial if the smaller operand is used in other costly operations.
    * **Efficient TLU**: requires at most two TLUs.
  * **Cons:**
    * **Slowdowns for bigger operand**: increasing the bit width of the bigger operand can slow down other costly operations.
    * **Potential inefficiency**: If you don't use the smaller operands for other operations, or do less costly operations than comparisons, it could introduce an unnecessary TLU and slow down execution compared to `fhe.BitwiseStrategy.THREE_TLU_CASTED`.

## 4. Two TLUs: bigger cast and smaller promoted

### How it works

This method constrains only the smaller operand to have at least the bid width to store `pack(x, y)` and casts the bigger operand to that bid width during runtime. It works like this:

```python
uint6_to_uint9_lut = fhe.LookupTable([...])
y_cast_to_uint9 = uint6_to_uint9_lut[y]

comparison_lut = fhe.LookupTable([...])
result = comparison_lut[x_promoted_to_uint9 - y_cast_to_uint9]
```

### How to use in FHE

The following example demonstrates how to implement subtraction with 2 TLUs (bigger cast and smaller promoted) using **Concrete** to perform bitwise AND operations over two encrypted integers:

```python
import numpy as np
from concrete import fhe

configuration = fhe.Configuration(
    bitwise_strategy_preference=fhe.BitwiseStrategy.TWO_TLU_BIGGER_CASTED_SMALLER_PROMOTED,
)

def f(x, y):
    return x | y

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
  func.func @main(%arg0: !FHE.eint<9>, %arg1: !FHE.eint<6>) -> !FHE.eint<6> {
    
    // casting bigger operand
    %cst = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]> : tensor<64xi64>
    %0 = "FHE.apply_lookup_table"(%arg1, %cst) : (!FHE.eint<6>, tensor<64xi64>) -> !FHE.eint<9>
        
    // packing
    %c64_i10 = arith.constant 64 : i10
    %1 = "FHE.mul_eint_int"(%arg0, %c64_i10) : (!FHE.eint<9>, i10) -> !FHE.eint<9>
    %2 = "FHE.add_eint"(%1, %0) : (!FHE.eint<9>, !FHE.eint<9>) -> !FHE.eint<9>
        
    // computing the result
    %cst_0 = arith.constant dense<"..."> : tensor<512xi64>
    %3 = "FHE.apply_lookup_table"(%2, %cst_0) : (!FHE.eint<9>, tensor<512xi64>) -> !FHE.eint<6>
        
    return %3 : !FHE.eint<6>

  }
  
}
```

</details>

### Considerations

* **The number of TLU:** if the bigger operand is assigned the same bit width as the smaller operand because of other operations, this method can use only a single TLU.
* **Pros and cons:**
  * **Pros:**
    * **Minimal constraints**: only the smaller operand is constrained, which is beneficial if the bigger operand is used in other costly operations.
    * **Efficient TLU**: requires at most two TLUs.
  * **Cons:**
    * **Slowdowns for bigger operand**: increasing the bit-width of the smaller operand can slow down other costly operations.
    * **Potential inefficiency**: If you don't use the bigger operands for other operations, or do less costly operations than comparisons, it could introduce an unnecessary TLU and slow down execution compared to `fhe.BitwiseStrategy.THREE_TLU_CASTED`.
