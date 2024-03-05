<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.mlir.context`
Declaration of `Context` class. 

**Global Variables**
---------------
- **MAX_EXTRACTABLE_BIT**
- **MIN_EXTRACTABLE_BIT**
- **MAXIMUM_TLU_BIT_WIDTH**


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L44"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Context`
Context class, to perform operations on conversions. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L61"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(context: <locals>Context, graph: Graph, configuration: Configuration)
```








---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L1546"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add`

```python
add(resulting_type: ConversionType, x: Conversion, y: Conversion) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L1578"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `array`

```python
array(resulting_type: ConversionType, elements: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L1614"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `assign_static`

```python
assign_static(
    resulting_type: ConversionType,
    x: Conversion,
    y: Conversion,
    index: Sequence[Union[int, integer, slice]]
)
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L145"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `attribute`

```python
attribute(resulting_type: ConversionType, value: Any) → Attribute
```

Create an MLIR attribute. 



**Args:**
  resulting_type (ConversionType):  type of the attribute 

 value (Any):  value of the attribute 



**Returns:**
  MlirAttribute:  resulting MLIR attribute 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L827"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `best_chunk_ranges`

```python
best_chunk_ranges(
    x: Conversion,
    x_offset: int,
    y: Conversion,
    y_offset: int
) → List[Tuple[int, int]]
```

Calculate best chunk ranges for given operands. 



**Args:**
  x (Conversion)  lhs of the operation 

 x_offset (int)  lhs offset 

 y (Conversion)  rhs of the operation 

 y_offset (int)  rhs offset 



**Returns:**
  List[Tuple[int, int]]:  best chunk ranges for the arguments 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L1707"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `bitwise`

```python
bitwise(
    resulting_type: ConversionType,
    x: Conversion,
    y: Conversion,
    operation: Callable[[int, int], int]
) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L1854"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `bitwise_and`

```python
bitwise_and(
    resulting_type: ConversionType,
    x: Conversion,
    y: Conversion
) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L1862"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `bitwise_or`

```python
bitwise_or(
    resulting_type: ConversionType,
    x: Conversion,
    y: Conversion
) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L1870"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `bitwise_xor`

```python
bitwise_xor(
    resulting_type: ConversionType,
    x: Conversion,
    y: Conversion
) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L1878"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `broadcast_to`

```python
broadcast_to(x: Conversion, shape: Tuple[int, ])
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L1899"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `cast`

```python
cast(resulting_type: ConversionType, x: Conversion) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L370"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compare_with_subtraction`

```python
compare_with_subtraction(
    resulting_type: ConversionType,
    subtraction: Conversion,
    accept: Set[Comparison]
) → Conversion
```

Apply the final comparison table and return comparison result. 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L265"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `comparison`

```python
comparison(
    resulting_type: ConversionType,
    x: Conversion,
    y: Conversion,
    accept: Set[Comparison]
) → Conversion
```

Compare two encrypted values. 



**Args:**
  resulting_type (ConversionType):  resulting type 

 x (Conversion):  lhs of comparison 

 y (Conversion):  rhs of comparison 

 accept (Set[Comparison]):  set of accepted comparison outcomes 



**Returns:**
  Conversion:  result of comparison 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L586"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `comparison_with_chunks`

```python
comparison_with_chunks(
    resulting_type: ConversionType,
    x: Conversion,
    y: Conversion,
    accept: Set[Comparison]
) → Conversion
```

Compare encrypted values using chunks. 

Idea:  split x and y into small chunks  compare the chunks using table lookups  reduce chunk comparisons to a final result 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L763"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `comparison_with_chunks_equals`

```python
comparison_with_chunks_equals(
    resulting_type: ConversionType,
    x: Conversion,
    y: Conversion,
    accept: Set[Comparison],
    x_offset: int,
    y_offset: int,
    x_was_signed: bool,
    y_was_signed: bool,
    chunk_ranges: List[Tuple[int, int]]
) → Conversion
```

Check equality of encrypted values using chunks. 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L406"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `comparison_with_subtraction_trick`

```python
comparison_with_subtraction_trick(
    resulting_type: ConversionType,
    x: Conversion,
    y: Conversion,
    accept: Set[Comparison],
    x_minus_y_dtype: Integer
) → Conversion
```

Compare encrypted values using subtraction trick. 

Idea:  x [.] y <==> (x - y) [.] 0 where [.] is one of <,<=,==,!=,>=,> 

Additional Args:  x_minus_y_dtype (Integer):  minimal dtype that can be used to store x - y without overflows 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L1915"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `concatenate`

```python
concatenate(
    resulting_type: ConversionType,
    xs: List[Conversion],
    axis: Optional[int]
) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L1962"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `constant`

```python
constant(resulting_type: ConversionType, data: Any) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L1985"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `conv2d`

```python
conv2d(
    resulting_type: ConversionType,
    x: Conversion,
    weight: Conversion,
    bias: Optional[Conversion],
    strides: Sequence[int],
    dilations: Sequence[int],
    pads: Sequence[int],
    group: int
)
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L888"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `convert_to_chunks_and_map`

```python
convert_to_chunks_and_map(
    resulting_scalar_type: ConversionType,
    resulting_shape: Tuple[int, ],
    chunk_ranges: List[Tuple[int, int]],
    x: Conversion,
    x_offset: int,
    y: Conversion,
    y_offset: int,
    mapper: Callable
) → List[Conversion]
```

Extract the chunks of two values, pack them in a single integer and map the integer. 



**Args:**
  resulting_scalar_type (ConversionType):  scalar type of the results 

 resulting_shape (ConversionType):  shape of the output of the operation 

 chunk_ranges (List[Tuple[int, int]]):  chunks ranges for the operation 

 x (Conversion):  first operand 

 y (Conversion):  second operand 

 mapper (Callable):  mapping function 

 x_offset (int, default = 0):  optional offset for x during chunk extraction 

 y_offset (int, default = 0):  optional offset for x during chunk extraction 



**Returns:**
  List[Conversion]:  result of mapping chunks of x and y 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L2050"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dot`

```python
dot(resulting_type: ConversionType, x: Conversion, y: Conversion) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L2123"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dynamic_tlu`

```python
dynamic_tlu(
    resulting_type: ConversionType,
    on: Conversion,
    table: Conversion
) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L81"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `eint`

```python
eint(width: int) → ConversionType
```

Get encrypted unsigned integer type (e.g., !FHE.eint<3>, !FHE.eint<5>). 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L2158"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `encrypt`

```python
encrypt(resulting_type: ConversionType, x: Conversion) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L2165"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `equal`

```python
equal(resulting_type: ConversionType, x: Conversion, y: Conversion) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L170"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `error`

```python
error(highlights: Mapping[Node, Union[str, List[str]]])
```

Fail compilation with an error. 



**Args:**
  highlights (Mapping[Node, Union[str, List[str]]]):  nodes to highlight along with messages 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L87"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `esint`

```python
esint(width: int) → ConversionType
```

Get encrypted signed integer type (e.g., !FHE.esint<3>, !FHE.esint<5>). 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L2168"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `extract_bits`

```python
extract_bits(
    resulting_type: ConversionType,
    x: Conversion,
    bits: Union[int, integer, slice]
) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L2258"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `flatten`

```python
flatten(x: Conversion) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L2261"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `greater`

```python
greater(
    resulting_type: ConversionType,
    x: Conversion,
    y: Conversion
) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L2264"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `greater_equal`

```python
greater_equal(
    resulting_type: ConversionType,
    x: Conversion,
    y: Conversion
) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L75"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `i`

```python
i(width: int) → ConversionType
```

Get clear signless integer type (e.g., i3, i5). 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L2272"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `index_static`

```python
index_static(
    resulting_type: ConversionType,
    x: Conversion,
    index: Sequence[Union[int, integer, slice, ndarray, list]]
) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L2407"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `index_static_fancy`

```python
index_static_fancy(
    resulting_type: ConversionType,
    x: Conversion,
    index: Sequence[Union[int, integer, slice, ndarray, list]]
) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L93"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `index_type`

```python
index_type() → Type
```

Get index type. 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L181"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_bit_width_compatible`

```python
is_bit_width_compatible(*args: Optional[ConversionType, Conversion]) → bool
```

Check if conversion types are compatible in terms of bit-width. 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L2439"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `less`

```python
less(resulting_type: ConversionType, x: Conversion, y: Conversion) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L2442"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `less_equal`

```python
less_equal(
    resulting_type: ConversionType,
    x: Conversion,
    y: Conversion
) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L130"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `location`

```python
location() → Location
```

Create an MLIR location from the node that is being converted. 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L2450"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `lsb`

```python
lsb(resulting_type: ConversionType, x: Conversion) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L2457"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `matmul`

```python
matmul(
    resulting_type: ConversionType,
    x: Conversion,
    y: Conversion
) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L2531"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `maximum`

```python
maximum(
    resulting_type: ConversionType,
    x: Conversion,
    y: Conversion
) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L2604"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `maxpool2d`

```python
maxpool2d(
    resulting_type: ConversionType,
    x: Conversion,
    kernel_shape: Tuple[int, ],
    strides: Sequence[int],
    dilations: Sequence[int]
)
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L2653"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `minimum`

```python
minimum(
    resulting_type: ConversionType,
    x: Conversion,
    y: Conversion
) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L1170"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `minimum_maximum_with_chunks`

```python
minimum_maximum_with_chunks(
    resulting_type: ConversionType,
    x: Conversion,
    y: Conversion,
    operation: str
) → Conversion
```

Calculate minimum or maximum between two encrypted values using chunks. 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L1110"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `minimum_maximum_with_trick`

```python
minimum_maximum_with_trick(
    resulting_type: ConversionType,
    x: Conversion,
    y: Conversion,
    x_minus_y_dtype: Integer,
    intermediate_table: List[int]
) → Conversion
```

Calculate minimum or maximum between two encrypted values using minimum or maximum trick. 

Idea:  min(x, y) <==> min(x - y, 0) + y  max(x, y) <==> max(x - y, 0) + y 

Additional Args:  x_minus_y_dtype (Integer):  minimal dtype that can be used to store x - y without overflows 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L2727"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `mul`

```python
mul(resulting_type: ConversionType, x: Conversion, y: Conversion) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L2812"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `multi_tlu`

```python
multi_tlu(
    resulting_type: ConversionType,
    on: Conversion,
    tables: Any,
    mapping: Any
)
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L1411"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `multiplication_with_boolean`

```python
multiplication_with_boolean(
    boolean: Conversion,
    value: Conversion,
    resulting_bit_width: int,
    chunk_size: int,
    inverted: bool = False
)
```

Calculate boolean * value using bits. 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L2865"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `multivariate_multi_tlu`

```python
multivariate_multi_tlu(
    resulting_type: ConversionType,
    xs: List[Conversion],
    tables: Any,
    mapping: Any
)
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L2854"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `multivariate_tlu`

```python
multivariate_tlu(
    resulting_type: ConversionType,
    xs: List[Conversion],
    table: Sequence[int]
) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L2877"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `neg`

```python
neg(resulting_type: ConversionType, x: Conversion) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L2895"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `not_equal`

```python
not_equal(
    resulting_type: ConversionType,
    x: Conversion,
    y: Conversion
) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L2903"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ones`

```python
ones(resulting_type: ConversionType) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L212"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `operation`

```python
operation(
    operation: Callable,
    resulting_type: ConversionType,
    *args,
    original_bit_width: Optional[int] = None,
    **kwargs
) → Conversion
```

Create a conversion from an MLIR operation. 



**Args:**
  operation (Callable):  MLIR operation to create (e.g., fhe.AddEintOp) 

 resulting_type (ConversionType):  type of the output of the operation 

 *args (Any):  args to pass to the operation 

 original_bit_width (Optional[int], default = None):  original bit width of the resulting conversion 

 *kwargs (Any):  kwargs to pass to the operation 

**Returns:**
  Conversion:  resulting conversion 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L1028"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `pack_multivariate_inputs`

```python
pack_multivariate_inputs(xs: List[Conversion]) → Conversion
```

Packs inputs of multivariate table lookups. 



**Args:**
  xs (List[Conversion]):  operands 



**Returns:**
  Conversion:  packed operands 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L3565"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `reinterpret`

```python
reinterpret(x: Conversion, bit_width: int) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L2917"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `relu`

```python
relu(resulting_type: ConversionType, x: Conversion) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L2973"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `reshape`

```python
reshape(x: Conversion, shape: Tuple[int, ]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L3110"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `round_bit_pattern`

```python
round_bit_pattern(
    resulting_type: ConversionType,
    x: Conversion,
    lsbs_to_remove: int
) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L3141"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `shift`

```python
shift(
    resulting_type: ConversionType,
    x: Conversion,
    b: Conversion,
    orientation: str,
    original_resulting_bit_width: int
) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L3342"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `sub`

```python
sub(resulting_type: ConversionType, x: Conversion, y: Conversion) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L3375"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `sum`

```python
sum(
    resulting_type: ConversionType,
    x: Conversion,
    axes: Optional[int, Sequence[int]] = (),
    keep_dims: bool = False
) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L99"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `tensor`

```python
tensor(element_type: ConversionType, shape: Tuple[int, ]) → ConversionType
```

Get tensor type (e.g., tensor<5xi3>, tensor<3x2x!FHE.eint<5>>). 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L3424"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `tensorize`

```python
tensorize(x: Conversion) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L3440"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `tlu`

```python
tlu(resulting_type: ConversionType, on: Conversion, table: Sequence[int])
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L3469"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_signed`

```python
to_signed(x: Conversion) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L3491"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_signedness`

```python
to_signedness(x: Conversion, of: ConversionType) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L3494"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_unsigned`

```python
to_unsigned(x: Conversion) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L3516"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `transpose`

```python
transpose(
    resulting_type: ConversionType,
    x: Conversion,
    axes: Sequence[int] = ()
)
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L3527"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `tree_add`

```python
tree_add(resulting_type: ConversionType, xs: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L3545"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `truncate_bit_pattern`

```python
truncate_bit_pattern(x: Conversion, lsbs_to_remove: int) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L448"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `try_comparison_with_clipping_trick`

```python
try_comparison_with_clipping_trick(
    resulting_type: ConversionType,
    x: Conversion,
    y: Conversion,
    accept: Set[Comparison]
) → Optional[Conversion]
```

Compare encrypted values using clipping trick. 

Idea:  x [.] y <==> (clipped(x) - y) [.] 0 where [.] is one of <,<=,==,!=,>=,>  or  x [.] y <==> (x - clipped(y)) [.] 0 where [.] is one of <,<=,==,!=,>=,>  where  clipped(value) = np.clip(value, smaller.min() - 1, smaller.max() + 1) 

Additional Args:  smaller_minus_clipped_bigger_dtype (Integer):  minimal dtype that can be used to store smaller - clipped(bigger) without overflows 

 clipped_bigger_minus_smaller_dtype (Integer):  minimal dtype that can be used to store clipped(bigger) - smaller without overflows 

 smaller_bounds (Tuple[int, int]):  bounds of smaller 

 smaller_is_lhs (bool):  whether smaller is lhs of the comparison 

 smaller_is_rhs (bool):  whether smaller is rhs of the comparison 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L109"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `typeof`

```python
typeof(value: Union[ValueDescription, Node]) → ConversionType
```

Get type corresponding to a value or a node. 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L3579"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `where`

```python
where(
    resulting_type: ConversionType,
    condition: Conversion,
    when_true: Conversion,
    when_false: Conversion
) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/context.py#L3637"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `zeros`

```python
zeros(resulting_type: ConversionType) → Conversion
```






