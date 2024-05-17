# Chunk

This document explains the chunk strategy for comparison operations on encrypted data.

The chunk strategy breaks down the operands into smaller chunks and then processes chunks using table lookups (TLUs) with multiplications and additions. This general strategy is suitable for any situation.

## How it works

The following code explains how to compare two 8-bit operands by dividing them into 4-bit chunks:

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

## How to use in FHE

Here is how to implement the chunk method in **Concrete** to compare two encrypted integers:

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

<details>

<summary>The MLIR code generated</summary>

```
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

</details>

## Considerations

* **Signed comparison:** supported
* **Chunk size:** automatically selected by Concrete to optimize the number of TLUs.
* **Number of TLUs resulted:**  between 5 to 13
  * **Exception:** `==` and `!=` use a different chunk comparison and reduction strategy with fewer TLUs.
* **When to use:** when no other implementations can be used
* **Pros and cons**
  * **Pros:** compatible with any integers
  * **Cons:** very expensive
