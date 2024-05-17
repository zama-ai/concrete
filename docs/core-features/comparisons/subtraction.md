# Subtraction

This document explains the subtraction strategy for comparison operations and demonstrates four implementation methods.

## Introduction

The subtraction method converts `x [<,<=,==,!=,>=,>] y` to `x - y [<,<=,==,!=,>=,>] 0`, which is a simple subtraction and a Table Lookup (TLU).

### Requirements

* **Subtraction overflow**: to avoid overflow, the subtraction before the TLU requires up to two additional bits (one in most cases).
* **Uniform bit width:** the subtraction requires the operands to have the same bit width.
* **Maximum bit width:** the bit width required to store the result of the subtraction `x - y` must not exceed the maximum bit width the TLU can handle. In other words:
  * ```
    (x - y).bit_width <= MAXIMUM_TLU_BIT_WIDTH
    ```

For example, when comparing `uint3` and `uint6`, we need to convert both to `uint7` for the subtraction and proceed with the TLU in 7 bits.

There are 4 ways to implement subtraction.

## 1. One TLU promoted

### How it works

This method ensures that both operands are assigned the same bit width, which contains at least the number of bits required to store `x - y.` It works like this:

```python
comparison_lut = fhe.LookupTable([...])
result = comparison_lut[x_promoted_to_uint7 - y_promoted_to_uint7]
```

### How to use in FHE

The following example demonstrates how to implement subtraction with one TLU promoted using **Concrete** to compare two encrypted integers:

```python
import numpy as np
from concrete import fhe

configuration = fhe.Configuration(
    comparison_strategy_preference=fhe.ComparisonStrategy.ONE_TLU_PROMOTED,
)

def f(x, y):
    return x < y

inputset = [
    (np.random.randint(0, 2**4), np.random.randint(0, 2**4))
    for _ in range(100)
]

compiler = fhe.Compiler(f, {"x": "encrypted", "y": "encrypted"})
circuit = compiler.compile(inputset, configuration, show_mlir=True)
```

<details>

<summary>The MLIR code generated</summary>

```
module {
  // promotions          ............         ............
  func.func @main(%arg0: !FHE.eint<5>, %arg1: !FHE.eint<5>) -> !FHE.eint<1> {
    
    // subtraction
    %0 = "FHE.to_signed"(%arg0) : (!FHE.eint<5>) -> !FHE.esint<5>
    %1 = "FHE.to_signed"(%arg1) : (!FHE.eint<5>) -> !FHE.esint<5>
    %2 = "FHE.sub_eint"(%0, %1) : (!FHE.esint<5>, !FHE.esint<5>) -> !FHE.esint<5>
    
    // computing the result
    %cst = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]> : tensor<32xi64>
    %3 = "FHE.apply_lookup_table"(%2, %cst) : (!FHE.esint<5>, tensor<32xi64>) -> !FHE.eint<1>
    
    return %3 : !FHE.eint<1>
    
  }
  
}
```

</details>

### Considerations

* **Pros:**
  * **Efficient TLU:**  this method results in only one single TLU
* **Cons:**
  * **Potential slowdowns:** this method increases the bit width of both operands and locks them together across the whole circuit, potentially causing slowdowns in other costly operations.

## 2. Three TLUs cast

### How it works

This method does not constrain bit widths during bit-width assignment. It casts operands to a bit width that can store `x - y` during runtime using TLUs. It works like this:

```python
uint3_to_uint7_lut = fhe.LookupTable([...])
x_cast_to_uint7 = uint3_to_uint7_lut[x]

uint6_to_uint7_lut = fhe.LookupTable([...])
y_cast_to_uint7 = uint6_to_uint7_lut[y]

comparison_lut = fhe.LookupTable([...])
result = comparison_lut[x_cast_to_uint7 - y_cast_to_uint7]
```

### How to use in FHE

The following example demonstrates how to implement subtraction with three TLUs cast using **Concrete** to compare two encrypted integers:

```python
import numpy as np
from concrete import fhe

configuration = fhe.Configuration(
    comparison_strategy_preference=fhe.ComparisonStrategy.THREE_TLU_CASTED,
)

def f(x, y):
    return x < y

inputset = [
    (np.random.randint(0, 2**4), np.random.randint(0, 2**4))
    for _ in range(100)
]

compiler = fhe.Compiler(f, {"x": "encrypted", "y": "encrypted"})
circuit = compiler.compile(inputset, configuration, show_mlir=True)
```

<details>

<summary>The MLIR code generated</summary>

```
module {
  
  // no promotions
  func.func @main(%arg0: !FHE.eint<3>, %arg1: !FHE.eint<6>) -> !FHE.eint<1> {
    
    // casting
    %cst = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : tensor<16xi64>
    %0 = "FHE.apply_lookup_table"(%arg0, %cst) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.esint<5>
    %1 = "FHE.apply_lookup_table"(%arg1, %cst) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.esint<5>
    
    // subtraction
    %2 = "FHE.sub_eint"(%0, %1) : (!FHE.esint<5>, !FHE.esint<5>) -> !FHE.esint<5>
    
    // computing the result
    %cst_0 = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]> : tensor<32xi64>
    %3 = "FHE.apply_lookup_table"(%2, %cst_0) : (!FHE.esint<5>, tensor<32xi64>) -> !FHE.eint<1>
    
    return %3 : !FHE.eint<1>
    
  }
  
}
```

</details>

### Considerations

* **Number of TLUs:**
  * If `x` and `y` are assigned the same bit width which can store `x - y` because of other operations, this method uses only a single TLU.
  * If one of the operands is assigned a bit width bigger than or equal to the required bit width to store `x - y,` this method can use 2 TLUs.
* **Pros and cons:**
  * **Pros:**
    * **No bit width constraints**: this method does not constraint the bit widths of the operands, which is beneficial when other costly operations are involved.
    * **Efficient TLU**: it requires at most three TLUs, which is still efficient.
  * **Cons:**
    * **Potential inefficiency:** if you don't use the operands for other operations, or do less costly operations than comparisons, this method introduces up to two unnecessary TLUs and slows down execution compared to `fhe.ComparisonStrategy.ONE_TLU_PROMOTED`.

## 3. Two TLUs: bigger promoted and smaller cast

### How it works

This method constrains only the bigger operand to have at least the bid width to store `x - y` and casts the smaller operand to that bid width during runtime. It works like this:

```python
uint3_to_uint7_lut = fhe.LookupTable([...])
x_cast_to_uint7 = uint3_to_uint7_lut[x]

comparison_lut = fhe.LookupTable([...])
result = comparison_lut[x_cast_to_uint7 - y_promoted_to_uint7]
```

### How to use in FHE

The following example demonstrates how to implement subtraction with 2 TLUs (bigger promoted and smaller cast) using **Concrete** to compare two encrypted integers:

```python
import numpy as np
from concrete import fhe

configuration = fhe.Configuration(
    comparison_strategy_preference=fhe.ComparisonStrategy.TWO_TLU_BIGGER_PROMOTED_SMALLER_CASTED,
)

def f(x, y):
    return x < y

inputset = [
    (np.random.randint(0, 2**3), np.random.randint(0, 2**5))
    for _ in range(100)
]

compiler = fhe.Compiler(f, {"x": "encrypted", "y": "encrypted"})
circuit = compiler.compile(inputset, configuration, show_mlir=True)
```

<details>

<summary>The MLIR code generated</summary>

```
module {
  
  // promotions                               ............
  func.func @main(%arg0: !FHE.eint<3>, %arg1: !FHE.eint<6>) -> !FHE.eint<1> {
    
    // casting the smaller operand
    %cst = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi64>
    %0 = "FHE.apply_lookup_table"(%arg0, %cst) : (!FHE.eint<3>, tensor<8xi64>) -> !FHE.esint<6>
    
    // subtraction
    %1 = "FHE.to_signed"(%arg1) : (!FHE.eint<6>) -> !FHE.esint<6>
    %2 = "FHE.sub_eint"(%0, %1) : (!FHE.esint<6>, !FHE.esint<6>) -> !FHE.esint<6>
    
    // computing the result
    %cst_0 = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]> : tensor<64xi64>
    %3 = "FHE.apply_lookup_table"(%2, %cst_0) : (!FHE.esint<6>, tensor<64xi64>) -> !FHE.eint<1>
    
    return %3 : !FHE.eint<1>
    
  }
  
}
```

</details>

### Considerations

* **The number of TLU:** if the smaller operand is assigned the same bit width as the bigger operand because of other operations, this method uses only a single TLU.
* **Pros and cons:**
  * **Pros:**
    * **Minimal constraints**: only the bigger operand is constrained, which is beneficial if the smaller operand is used in other costly operations.
    * **Efficient TLU**: requires at most two TLUs.
  * **Cons:**
    * **Slowdowns for bigger operand**: increasing the bit-width of the bigger operand can slow down other costly operations.
    * **Potential inefficiency**: if you don't use the smaller operands for other operations, or do less costly operations than comparisons, it could introduce an unnecessary TLU and slow down execution compared to `fhe.ComparisonStrategy.THREE_TLU_CASTED`.

## 4. Two TLUs: bigger cast and smaller promoted

### How it works

This method constrains only the smaller operand to have at least the bid width to store `x - y` and casts the bigger operand to that bid width during runtime. It works like this:

```python
uint6_to_uint7_lut = fhe.LookupTable([...])
y_cast_to_uint7 = uint6_to_uint7_lut[y]

comparison_lut = fhe.LookupTable([...])
result = comparison_lut[x_promoted_to_uint7 - y_cast_to_uint7]
```

### How to use in FHE

The following example demonstrates how to implement subtraction with 2 TLUs (bigger cast and smaller promoted) using **Concrete** to compare two encrypted integers:

```python
import numpy as np
from concrete import fhe

configuration = fhe.Configuration(
    comparison_strategy_preference=fhe.ComparisonStrategy.TWO_TLU_BIGGER_PROMOTED_SMALLER_CASTED,
)

def f(x, y):
    return x < y

inputset = [
    (np.random.randint(0, 2**3), np.random.randint(0, 2**5))
    for _ in range(100)
]

compiler = fhe.Compiler(f, {"x": "encrypted", "y": "encrypted"})
circuit = compiler.compile(inputset, configuration, show_mlir=True)
```

<details>

<summary>The MLIR code generated</summary>

```
module {
  
  // promotions          ............
  func.func @main(%arg0: !FHE.eint<6>, %arg1: !FHE.eint<5>) -> !FHE.eint<1> {
    
    // casting the bigger operand
    %cst = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]> : tensor<32xi64>
    %0 = "FHE.apply_lookup_table"(%arg1, %cst) : (!FHE.eint<5>, tensor<32xi64>) -> !FHE.esint<6>
    
    // subtraction
    %1 = "FHE.to_signed"(%arg0) : (!FHE.eint<6>) -> !FHE.esint<6>
    %2 = "FHE.sub_eint"(%1, %0) : (!FHE.esint<6>, !FHE.esint<6>) -> !FHE.esint<6>
    
    // computing the result
    %cst_0 = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]> : tensor<64xi64>
    %3 = "FHE.apply_lookup_table"(%2, %cst_0) : (!FHE.esint<6>, tensor<64xi64>) -> !FHE.eint<1>
    
    return %3 : !FHE.eint<1>
    
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
    * **Potential inefficiency**: if you don't use the bigger operands for other operations, or do less costly operations than comparisons, it could introduce an unnecessary TLU and slow down execution compared to `fhe.ComparisonStrategy.THREE_TLU_CASTED`.
