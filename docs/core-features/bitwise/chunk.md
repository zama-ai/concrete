# Chunk

This document explains the chunk strategy for bitwise operations on encrypted data.

The chunk strategy breaks down the operands into smaller chunks and then processes chunks using Table Lookups (TLUs) with multiplications and additions. This general strategy is suitable for any situation.

## How it works

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

## How to use in FHE

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

<details>

<summary>The MLIR code generated</summary>

```cpp
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

</details>

### Considerations

* **Signed bitwise:** not supported.
* **Chunk size:  Concrete** automatically selects the optimal chunk size to reduce the number of TLUs.
* **Number of TLUs:** between 4 to 9 TLUs.
* **When to use:** when no other implementations can be used
* **Pros and cons**
  * **Pros:** compatible with any integers
  * **Cons:** very expensive
