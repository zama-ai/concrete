# Bitwise Operations

This document describes how comparisons are managed in Concrete, typically "AND", "OR", and so on. It covers different strategies to make the FHE computations faster, depending on the context.

Bitwise operations are not native operations in Concrete, so they need to be implemented using existing native operations (i.e., additions, clear multiplications, negations, table lookups). Concrete offers two different implementations for performing bitwise operations.

## Chunked

This is the most general implementation that can be used in any situation. The idea is:

```python
# (example below is for bit-width of 8 and chunk size of 4)

# extract chunks of lhs using table lookups
lhs_chunks = [lhs.bits[0:4], lhs.bits[4:8]]

# extract chunks of rhs using table lookups
rhs_chunks = [rhs.bits[0:4], rhs.bits[4:8]]

# pack chunks of lhs and rhs using clear multiplications and additions 
packed_chunks = []
for lhs_chunk, rhs_chunk in zip(lhs_chunks, rhs_chunks):
    shifted_lhs_chunk = lhs_chunk * 2**4  # (i.e., lhs_chunk << 4)
    packed_chunks.append(shifted_lhs_chunk + rhs_chunk)

# apply comparison table lookup to packed chunks
bitwise_table = fhe.LookupTable([...])
result_chunks = bitwise_table[packed_chunks]

# sum resulting chunks obtain the result
result = np.sum(result_chunks)
```

### Notes

- Signed bitwise operations are not supported.
- The optimal chunk size is selected automatically to reduce the number of table lookups.
- Chunked bitwise operations result in at least 4 and at most 9 table lookups.
- It is used if no other implementation can be used.

### Pros

- Can be used with any integers.

### Cons

- Very expensive.

### Example

```python
import numpy as np
from concrete import fhe

def f(x, y):
    return x & y

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
  
  // no promotions
  func.func @main(%arg0: !FHE.eint<4>, %arg1: !FHE.eint<4>) -> !FHE.eint<4> {

    // extracting the first chunk of x, adjusted for shifting
    %cst = arith.constant dense<[0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12]> : tensor<16xi64>
    %0 = "FHE.apply_lookup_table"(%arg0, %cst) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.eint<4>
        
    // extracting the first chunk of y
    %cst_0 = arith.constant dense<[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]> : tensor<16xi64>
    %1 = "FHE.apply_lookup_table"(%arg1, %cst_0) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.eint<4>
        
    // packing the first chunks
    %2 = "FHE.add_eint"(%0, %1) : (!FHE.eint<4>, !FHE.eint<4>) -> !FHE.eint<4>
        
    // applying the bitwise operation to the first chunks, adjusted for addition in the end
    %cst_1 = arith.constant dense<[0, 0, 0, 0, 0, 4, 0, 4, 0, 0, 8, 8, 0, 4, 8, 12]> : tensor<16xi64>
    %3 = "FHE.apply_lookup_table"(%2, %cst_1) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.eint<4>
        
    // extracting the second chunk of x, adjusted for shifting
    %cst_2 = arith.constant dense<[0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12]> : tensor<16xi64>
    %4 = "FHE.apply_lookup_table"(%arg0, %cst_2) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.eint<4>
        
    // extracting the second chunk of y
    %cst_3 = arith.constant dense<[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]> : tensor<16xi64>
    %5 = "FHE.apply_lookup_table"(%arg1, %cst_3) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.eint<4>
        
    // packing the second chunks
    %6 = "FHE.add_eint"(%4, %5) : (!FHE.eint<4>, !FHE.eint<4>) -> !FHE.eint<4>
        
    // applying the bitwise operation to second chunks
    %cst_4 = arith.constant dense<[0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 2, 2, 0, 1, 2, 3]> : tensor<16xi64>
    %7 = "FHE.apply_lookup_table"(%6, %cst_4) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.eint<4>
        
    // adding resulting chunks to obtain the result
    %8 = "FHE.add_eint"(%7, %3) : (!FHE.eint<4>, !FHE.eint<4>) -> !FHE.eint<4>
        
    return %8 : !FHE.eint<4>

  }
  
}
```

## Packing Trick

This implementation uses the fact that we can combine two values into a single value and apply a single table lookup to this combined value!

There are two major problems with this implementation:
1) packing requires the same bit-width across operands.
2) packing requires the bit-width of at least `x.bit_width + y.bit_width` and that bit-width cannot exceed maximum TLU bit-width, which is `16` at the moment.

What this means is if we are comparing `uint3` and `uint6`, we need to convert both of them to `uint9` in some way to do the packing and proceed with the TLU in 9-bits. There are 4 ways to achieve this behavior.

### Requirements

- ```python
  x.bit_width + y.bit_width <= MAXIMUM_TLU_BIT_WIDTH
  ```

### 1. fhe.BitwiseStrategy.ONE_TLU_PROMOTED

This strategy makes sure that during bit-width assignment, both operands are assigned the same bit-width, and that bit-width contains at least the amount of bits required to store `pack(x, y)`. The idea is:

```python
bitwise_lut = fhe.LookupTable([...])
result = bitwise_lut[pack(x_promoted_to_uint9, y_promoted_to_uint9)]
```

#### Pros

- It will always result in a single table lookup.

#### Cons

- It will significantly increase the bit-width of both operands and lock them to each other across the whole circuit, which can result in significant slowdowns if the operands are used in other costly operations.

#### Example

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

produces

```c++
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

### 2. fhe.BitwiseStrategy.THREE_TLU_CASTED

This strategy will not put any constraint on bit-widths during bit-width assignment, instead operands are cast to a bit-width that can store `pack(x, y)` during runtime using table lookups. The idea is:

```python
uint3_to_uint9_lut = fhe.LookupTable([...])
x_cast_to_uint9 = uint3_to_uint9_lut[x]

uint6_to_uint9_lut = fhe.LookupTable([...])
y_cast_to_uint9 = uint6_to_uint9_lut[y]

bitwise_lut = fhe.LookupTable([...])
result = bitwise_lut[pack(x_cast_to_uint9, y_cast_to_uint9)]
```

#### Notes

- It can result in a single table lookup as well, if x and y are assigned (because of other operations) the same bit-width, and that bit-width can store `pack(x, y)`.
- Or in two table lookups if only one of the operands is assigned a bit-width bigger than or equal to the bit width that can store `pack(x, y)`.

#### Pros

- It will not put any constraints on bit-widths of the operands, which is amazing if they are used in other costly operations.
- It will result in at most 3 table lookups, which is still good.

#### Cons

- If you are not doing anything else with the operands, or doing less costly operations compared to bitwise, it will introduce up to two unnecessary table lookups and slow down execution compared to `fhe.BitwiseStrategy.ONE_TLU_PROMOTED`.

#### Example

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

produces

```c++
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

### 3. fhe.BitwiseStrategy.TWO_TLU_BIGGER_PROMOTED_SMALLER_CASTED 

This strategy can be viewed as a middle ground between the two strategies described above. With this strategy, only the bigger operand will be constrained to have at least the required bit-width to store `pack(x, y)`, and the smaller operand will be cast to that bit-width during runtime. The idea is:

```python
uint3_to_uint9_lut = fhe.LookupTable([...])
x_cast_to_uint9 = uint3_to_uint9_lut[x]

comparison_lut = fhe.LookupTable([...])
result = comparison_lut[x_cast_to_uint9 - y_promoted_to_uint9]
```

#### Notes

- It can result in a single table lookup as well, if the smaller operand is assigned (because of other operations) the same bit-width as the bigger operand.

#### Pros

- It will only put a constraint on the bigger operand, which is great if the smaller operand is used in other costly operations.
- It will result in at most 2 table lookups, which is great.

#### Cons

- It will significantly increase the bit-width of the bigger operand which can result in significant slowdowns if the bigger operand is used in other costly operations.
- If you are not doing anything else with the smaller operand, or doing less costly operations compared to comparison, it could introduce an unnecessary table lookup and slow down execution compared to `fhe.BitwiseStrategy.THREE_TLU_CASTED`.

#### Example

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

produces

```c++
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

### 4. fhe.BitwiseStrategy.TWO_TLU_BIGGER_CASTED_SMALLER_PROMOTED

This strategy is like the exact opposite of the strategy above. With this, only the smaller operand will be constrained to have at least the required bit-width, and the bigger operand will be cast during runtime. The idea is:

```python
uint6_to_uint9_lut = fhe.LookupTable([...])
y_cast_to_uint9 = uint6_to_uint9_lut[y]

comparison_lut = fhe.LookupTable([...])
result = comparison_lut[x_promoted_to_uint9 - y_cast_to_uint9]
```

#### Notes

- It can result in a single table lookup as well, if the bigger operand is assigned (because of other operations) the same bit-width as the smaller operand.

#### Pros

- It will only put constraint on the smaller operand, which is great if the bigger operand is used in other costly operations.
- It will result in at most 2 table lookups, which is great.

#### Cons

- It will increase the bit-width of the smaller operand which can result in significant slowdowns if the smaller operand is used in other costly operations.
- If you are not doing anything else with the bigger operand, or doing less costly operations compared to comparison, it could introduce an unnecessary table lookup and slow down execution compared to `fhe.BitwiseStrategy.THREE_TLU_CASTED`.

#### Example

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

produces

```c++
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

## Summary

| Strategy                                | Minimum # of TLUs | Maximum # of TLUs | Can increase the bit-width of the inputs |
|-----------------------------------------|-------------------|-------------------|------------------------------------------|
| CHUNKED                                 | 4                 | 9                 |                                          |
| ONE_TLU_PROMOTED                        | 1                 | 1                 | ✓                                        |
| THREE_TLU_CASTED                        | 1                 | 3                 |                                          |
| TWO_TLU_BIGGER_PROMOTED_SMALLER_CASTED  | 1                 | 2                 | ✓                                        |
| TWO_TLU_BIGGER_CASTED_SMALLER_PROMOTED  | 1                 | 2                 | ✓                                        |

{% hint style="info" %}
Concrete will choose the best strategy available after bit-width assignment, regardless of the specified preference.
{% endhint %}

{% hint style="info" %}
Different strategies are good for different circuits. If you want the best runtime for your use case, you can compile your circuit with all different comparison strategy preferences, and pick the one with the lowest complexity.
{% endhint %}

## Shifts

The same configuration option is used to modify the behavior of encrypted shift operations, and shifts are much more complex to implement, so we'll not go over the details. What is important is, that the end result is computed using additions or subtractions on the original shifted operand. Since additions and subtractions require the same bit-width across operands, input and output bit-widths need to be synchronized at some point. There are two ways to do this:

### With promotion

Here, the shifted operand and shift result are assigned the same bit-width during bit-width assignment, which avoids an additional TLU on the shifted operand. On the other hand, it might increase the bit-width of the result or the shifted operand, and if they're used in other costly operations, it could result in significant slowdowns. This is the default behavior.

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

produces

```c++
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

### With casting

The approach described above could be suboptimal for some circuits, so it is advised to check the complexity with it disabled before production. Here is how the implementation changes with it disabled.

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

produces

```c++
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
