# Comparisons

This document describes how comparisons are managed in Concrete, typically 'equal', 'greater than', and so on.  It covers different strategies to make the FHE computations faster, depending on the context.

Comparisons are not native operations in Concrete, so they need to be implemented using existing native operations (i.e., additions, clear multiplications, negations, table lookups). Concrete offers three different implementations for performing comparisons.

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
comparison_table = fhe.LookupTable([...])
chunk_comparisons = comparison_table[packed_chunks]

# reduce chunk comparisons to comparison of numbers
result = chunk_comparisons[0]
for chunk_comparison in chunk_comparisons[1:]:
    chunk_reduction_table = fhe.LookupTable([...])
    shifted_chunk_comparison= chunk_comparison * 2**2  # (i.e., lhs_chunk << 2)
    result = chunk_reduction_table[result + shifted_chunk_comparison]
```

### Notes

- Signed comparisons are more complex to explain, but they are supported!
- The optimal chunk size is selected automatically to reduce the number of table lookups.
- Chunked comparisons result in at least 5 and at most 13 table lookups.
- It is used if no other implementation can be used.
- `==` and `!=` are using a different chunk comparison and reduction strategy with less table lookups.

### Pros

- Can be used with any integers.

### Cons

- Very expensive.

### Example

```python
import numpy as np
from concrete import fhe

def f(x, y):
    return x < y

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
  func.func @main(%arg0: !FHE.eint<4>, %arg1: !FHE.eint<4>) -> !FHE.eint<1> {
  
    // extracting the first chunk of x, adjusted for shifting
    %cst = arith.constant dense<[0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12]> : tensor<16xi64>
    %0 = "FHE.apply_lookup_table"(%arg0, %cst) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.eint<4>
    
    // extracting the first chunk of y
    %cst_0 = arith.constant dense<[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]> : tensor<16xi64>
    %1 = "FHE.apply_lookup_table"(%arg1, %cst_0) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.eint<4>
    
    // packing first chunks
    %2 = "FHE.add_eint"(%0, %1) : (!FHE.eint<4>, !FHE.eint<4>) -> !FHE.eint<4>
    
    // comparing first chunks
    %cst_1 = arith.constant dense<[0, 1, 1, 1, 2, 0, 1, 1, 2, 2, 0, 1, 2, 2, 2, 0]> : tensor<16xi64>
    %3 = "FHE.apply_lookup_table"(%2, %cst_1) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.eint<4>
    
    // extracting the second chunk of x, adjusted for shifting
    %cst_2 = arith.constant dense<[0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12]> : tensor<16xi64>
    %4 = "FHE.apply_lookup_table"(%arg0, %cst_2) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.eint<4>
    
    // extracting the second chunk of y
    %cst_3 = arith.constant dense<[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]> : tensor<16xi64>
    %5 = "FHE.apply_lookup_table"(%arg1, %cst_3) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.eint<4>
    
    // packing second chunks
    %6 = "FHE.add_eint"(%4, %5) : (!FHE.eint<4>, !FHE.eint<4>) -> !FHE.eint<4>
    
    // comparing second chunks
    %cst_4 = arith.constant dense<[0, 4, 4, 4, 8, 0, 4, 4, 8, 8, 0, 4, 8, 8, 8, 0]> : tensor<16xi64>
    %7 = "FHE.apply_lookup_table"(%6, %cst_4) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.eint<4>
    
    // packing comparisons
    %8 = "FHE.add_eint"(%7, %3) : (!FHE.eint<4>, !FHE.eint<4>) -> !FHE.eint<4>
    
    // reducing comparisons to result
    %cst_5 = arith.constant dense<[0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]> : tensor<16xi64>
    %9 = "FHE.apply_lookup_table"(%8, %cst_5) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.eint<1>
    
    return %9 : !FHE.eint<1>
    
  }
}
```

## Subtraction Trick

This implementation uses the fact that `x [<,<=,==,!=,>=,>] y` is equal to `x - y [<,<=,==,!=,>=,>] 0`, which is just a subtraction and a table lookup!

There are two major problems with this implementation:
1) subtraction before the TLU requires up to 2 additional bits to avoid overflows (it is 1 in most cases).
2) subtraction requires the same bit-width across operands.

What this means is if we are comparing `uint3` and `uint6`, we need to convert both of them to `uint7` in some way to do the subtraction and proceed with the TLU in 7-bits. There are 4 ways to achieve this behavior.

### Requirements

- ```python
  (x - y).bit_width <= MAXIMUM_TLU_BIT_WIDTH
  ```

### 1. fhe.ComparisonStrategy.ONE_TLU_PROMOTED

This strategy makes sure that during bit-width assignment, both operands are assigned the same bit-width, and that bit-width contains at least the number of bits required to store `x - y`. The idea is:

```python
comparison_lut = fhe.LookupTable([...])
result = comparison_lut[x_promoted_to_uint7 - y_promoted_to_uint7]
```

#### Pros

- It will always result in a single table lookup.

#### Cons

- It will increase the bit-width of both operands and lock them to each other across the whole circuit, which can result in significant slowdowns if the operands are used in other costly operations.

#### Example

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

produces

```c++
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

### 2. fhe.ComparisonStrategy.THREE_TLU_CASTED

This strategy will not put any constraint on bit-widths during bit-width assignment, instead operands are cast to a bit-width that can store `x - y` during runtime using table lookups. The idea is:

```python
uint3_to_uint7_lut = fhe.LookupTable([...])
x_cast_to_uint7 = uint3_to_uint7_lut[x]

uint6_to_uint7_lut = fhe.LookupTable([...])
y_cast_to_uint7 = uint6_to_uint7_lut[y]

comparison_lut = fhe.LookupTable([...])
result = comparison_lut[x_cast_to_uint7 - y_cast_to_uint7]
```

#### Notes

- It can result in a single table lookup, if x and y are assigned (because of other operations) the same bit-width and that bit-width can store `x - y`.
- Alternatively, two table lookups can be used if only one of the operands is assigned a bit-width bigger than or equal to the bit width that can store `x - y`.

#### Pros

- It will not put any constraints on the bit-widths of the operands, which is amazing if they are used in other costly operations.
- It will result in at most 3 table lookups, which is still good.

#### Cons

- If you are not doing anything else with the operands, or doing less costly operations compared to comparison, it will introduce up to two unnecessary table lookups and slow down execution compared to `fhe.ComparisonStrategy.ONE_TLU_PROMOTED`.

#### Example

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

produces

```c++
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

### 3. fhe.ComparisonStrategy.TWO_TLU_BIGGER_PROMOTED_SMALLER_CASTED 

This strategy can be seen as a middle ground between the two strategies described above. With this strategy, only the bigger operand will be constrained to have at least the required bit-width to store `x - y`, and the smaller operand will be cast to that bit-width during runtime. The idea is:

```python
uint3_to_uint7_lut = fhe.LookupTable([...])
x_cast_to_uint7 = uint3_to_uint7_lut[x]

comparison_lut = fhe.LookupTable([...])
result = comparison_lut[x_cast_to_uint7 - y_promoted_to_uint7]
```

#### Notes

- It can result in a single table lookup, if the smaller operand is assigned (because of other operations) the same bit-width as the bigger operand.

#### Pros

- It will only put a constraint on the bigger operand, which is great if the smaller operand is used in other costly operations.
- It will result in at most 2 table lookups, which is great.

#### Cons

- It will increase the bit-width of the bigger operand, which can result in significant slowdowns if the bigger operand is used in other costly operations.
- If you are not doing anything else with the smaller operand, or doing less costly operations compared to comparison, it could introduce an unnecessary table lookup and slow down execution compared to `fhe.ComparisonStrategy.THREE_TLU_CASTED`.

#### Example

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

produces

```c++
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

### 4. fhe.ComparisonStrategy.TWO_TLU_BIGGER_CASTED_SMALLER_PROMOTED

This strategy can be seen as the exact opposite of the strategy above. With this, only the smaller operand will be constrained to have at least the required bit-width, and the bigger operand will be cast during runtime. The idea is:

```python
uint6_to_uint7_lut = fhe.LookupTable([...])
y_cast_to_uint7 = uint6_to_uint7_lut[y]

comparison_lut = fhe.LookupTable([...])
result = comparison_lut[x_promoted_to_uint7 - y_cast_to_uint7]
```

#### Notes

- It can result in a single table lookup, if the bigger operand is assigned (because of other operations) the same bit-width as the smaller operand.

#### Pros

- It will only put a constraint on the smaller operand, which is great if the bigger operand is used in other costly operations.
- It will result in at most 2 table lookups, which is great.

#### Cons

- It will increase the bit-width of the smaller operand, which can result in significant slowdowns if the smaller operand is used in other costly operations.
- If you are not doing anything else with the bigger operand, or doing less costly operations compared to comparison, it could introduce an unnecessary table lookup and slow down execution compared to `fhe.ComparisonStrategy.THREE_TLU_CASTED`.

#### Example

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

produces

```c++
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

## Clipping Trick

This implementation uses the fact that the subtraction trick is not optimal in terms of the required intermediate bit width. The comparison result does not change if we `compare(3, 40)` or `compare(3, 4)`, so why not clipping the bigger operand and then doing the subtraction to use less bits!

There are two major problems with this implementation:
1) it can not be used when the bit-widths are the same (for some cases even when they differ by only one bit)
2) subtraction still requires the same bit-width across operands.

What this means is if we are comparing `uint3` and `uint6`, we need to convert both of them to `uint4` in some way to do the subtraction and proceed with the TLU in 7-bits. There are 2 ways to achieve this behavior.

### Requirements

- ```python
  x.bit_width != y.bit_width
  ```
- ```python
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

### 1. fhe.ComparisonStrategy.THREE_TLU_BIGGER_CLIPPED_SMALLER_CASTED

This strategy will not put any constraint on bit-widths during bit-width assignment, instead the smaller operand is cast to a bit-width that can store `clipped(bigger) - smaller` or `smaller - clipped(bigger)` during runtime using table lookups. The idea is:

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

#### Notes

- This is a fallback implementation, so if there is a difference of 1-bit (or in some cases 2-bits) and the subtraction trick cannot be used optimally, this implementation will be used instead of `fhe.ComparisonStrategy.CHUNKED`.
- It can result in two table lookups if the smaller operand is assigned a bit-width bigger than or equal to the bit width that can store `clipped(bigger) - smaller` or `smaller - clipped(bigger)`.

#### Pros

- It will not put any constraints on the bit-widths of the operands, which is amazing if they are used in other costly operations.
- It will result in at most 3 table lookups, which is still good.
- These table lookups will be on smaller bit-widths, which is great.

#### Cons

- Cannot be used to compare integers with the same bit-width, which is very common.

#### Example

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

produces

```c++
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

### 2. fhe.ComparisonStrategy.TWO_TLU_BIGGER_CLIPPED_SMALLER_PROMOTED

This strategy is similar to the strategy described above. The difference is that with this strategy, the smaller operand will be constrained to have at least the required bit-width to store `clipped(bigger) - smaller` or `smaller - clipped(bigger)`. The bigger operand will still be clipped to that bit-width during runtime. The idea is:

```python
clipper = fhe.LookupTable([...])
y_clipped = clipper[y]

comparison_lut = fhe.LookupTable([...])
result = comparison_lut[x_promoted_to_uint4 - y_clipped]
# or
another_comparison_lut = fhe.LookupTable([...])
result = another_comparison_lut[y_clipped - x_promoted_to_uint4]
```

#### Pros

- It will only put a constraint on the smaller operand, which is great if the bigger operand is used in other costly operations.
- It will result in exactly 2 table lookups, which is great.

#### Cons

- It will increase the bit-width of the bigger operand, which can result in significant slowdowns if the bigger operand is used in other costly operations.

#### Example

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

produces

```c++
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

## Summary

| Strategy                                | Minimum # of TLUs | Maximum # of TLUs | Can increase the bit-width of the inputs |
|-----------------------------------------|-------------------|-------------------|------------------------------------------|
| CHUNKED                                 | 5                 | 13                |                                          |
| ONE_TLU_PROMOTED                        | 1                 | 1                 | ✓                                        |
| THREE_TLU_CASTED                        | 1                 | 3                 |                                          |
| TWO_TLU_BIGGER_PROMOTED_SMALLER_CASTED  | 1                 | 2                 | ✓                                        |
| TWO_TLU_BIGGER_CASTED_SMALLER_PROMOTED  | 1                 | 2                 | ✓                                        |
| THREE_TLU_BIGGER_CLIPPED_SMALLER_CASTED | 2                 | 3                 |                                          |
| TWO_TLU_BIGGER_CLIPPED_SMALLER_PROMOTED | 2                 | 2                 | ✓                                        |

{% hint style="info" %}
Concrete will choose the best strategy available after bit-width assignment, regardless of the specified preference.
{% endhint %}

{% hint style="info" %}
Different strategies are good for different circuits. If you want the best runtime for your use case, you can compile your circuit with all different comparison strategy preferences, and pick the one with the lowest complexity.
{% endhint %}
