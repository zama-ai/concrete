<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.mlir.context`
Declaration of `Context` class. 

**Global Variables**
---------------
- **MAX_EXTRACTABLE_BIT**
- **MIN_EXTRACTABLE_BIT**
- **MAXIMUM_TLU_BIT_WIDTH**
- **LUT_COSTS_V0_NORM2_0**


---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L69"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Context`
Context class, to perform operations on conversions. 

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L88"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(context: <locals>Context, graph: Graph, configuration: Configuration)
```








---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L1857"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add`

```python
add(resulting_type: ConversionType, x: Conversion, y: Conversion) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L1889"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `array`

```python
array(resulting_type: ConversionType, elements: list[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L1927"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `assign`

```python
assign(
    resulting_type: ConversionType,
    x: Conversion,
    y: Conversion,
    index: Sequence[Union[int, integer, slice, ndarray, list, Conversion]]
)
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L239"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L1121"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `best_chunk_ranges`

```python
best_chunk_ranges(
    x: Conversion,
    x_offset: int,
    y: Conversion,
    y_offset: int
) → list[tuple[int, int]]
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

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L1943"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L2090"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `bitwise_and`

```python
bitwise_and(
    resulting_type: ConversionType,
    x: Conversion,
    y: Conversion
) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L2098"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `bitwise_or`

```python
bitwise_or(
    resulting_type: ConversionType,
    x: Conversion,
    y: Conversion
) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L2106"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `bitwise_xor`

```python
bitwise_xor(
    resulting_type: ConversionType,
    x: Conversion,
    y: Conversion
) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L2114"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `broadcast_to`

```python
broadcast_to(x: Conversion, shape: tuple[int, ])
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L2127"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `cast`

```python
cast(resulting_type: ConversionType, x: Conversion) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L1826"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `cast_to_original_bit_width`

```python
cast_to_original_bit_width(value: Conversion) → Conversion
```

Cast a value to its original bit width using multiplication and reinterpretation. 

---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L4025"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `change_partition`

```python
change_partition(
    x: Conversion,
    src_partition: Optional[CryptoParams] = None,
    dest_partition: Optional[CryptoParams] = None
) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L664"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compare_with_subtraction`

```python
compare_with_subtraction(
    resulting_type: ConversionType,
    subtraction: Conversion,
    accept: set[Comparison]
) → Conversion
```

Apply the final comparison table and return comparison result. 

---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L559"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `comparison`

```python
comparison(
    resulting_type: ConversionType,
    x: Conversion,
    y: Conversion,
    accept: set[Comparison]
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

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L880"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `comparison_with_chunks`

```python
comparison_with_chunks(
    resulting_type: ConversionType,
    x: Conversion,
    y: Conversion,
    accept: set[Comparison]
) → Conversion
```

Compare encrypted values using chunks. 

Idea:  split x and y into small chunks  compare the chunks using table lookups  reduce chunk comparisons to a final result 

---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L1057"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `comparison_with_chunks_equals`

```python
comparison_with_chunks_equals(
    resulting_type: ConversionType,
    x: Conversion,
    y: Conversion,
    accept: set[Comparison],
    x_offset: int,
    y_offset: int,
    x_was_signed: bool,
    y_was_signed: bool,
    chunk_ranges: list[tuple[int, int]]
) → Conversion
```

Check equality of encrypted values using chunks. 

---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L700"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `comparison_with_subtraction_trick`

```python
comparison_with_subtraction_trick(
    resulting_type: ConversionType,
    x: Conversion,
    y: Conversion,
    accept: set[Comparison],
    x_minus_y_dtype: Integer
) → Conversion
```

Compare encrypted values using subtraction trick. 

Idea:  x [.] y <==> (x - y) [.] 0 where [.] is one of <,<=,==,!=,>=,> 

Additional Args:  x_minus_y_dtype (Integer):  minimal dtype that can be used to store x - y without overflows 

---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L2143"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `concatenate`

```python
concatenate(
    resulting_type: ConversionType,
    xs: list[Conversion],
    axis: Optional[int]
) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L384"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `conditional`

```python
conditional(
    resulting_type: Optional[ConversionType],
    condition: Conversion,
    then_builder: Callable[[], Optional[Conversion]],
    else_builder: Optional[Callable[[], Optional[Conversion]]] = None
) → Optional[Conversion]
```

Create an if conditional. 



**Args:**
  resulting_type (Optional[ConversionType]):  resulting type of the operation 

 condition (Conversion):  condition of conditional 

 then_builder (Callable[[], Optional[Conversion]]):  builder of then block of conditional 

 else_builder (Optional[Callable[[], Optional[Conversion]]], default = None):  optional builder of else block of conditional 



**Returns:**
  Optional[Conversion]:  None if resulting type is None  conversion of the created operation otherwise 



**Notes:**

> - if resulting type is not None both then and else builders need to return a conversion with the same type as resulting type 

---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L2185"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `constant`

```python
constant(
    resulting_type: ConversionType,
    data: Any,
    use_cache: bool = True
) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L2218"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L1185"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `convert_to_chunks_and_map`

```python
convert_to_chunks_and_map(
    resulting_scalar_type: ConversionType,
    resulting_shape: tuple[int, ],
    chunk_ranges: list[tuple[int, int]],
    x: Conversion,
    x_offset: int,
    y: Conversion,
    y_offset: int,
    mapper: Callable
) → list[Conversion]
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

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L2283"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dot`

```python
dot(resulting_type: ConversionType, x: Conversion, y: Conversion) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L2360"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dynamic_tlu`

```python
dynamic_tlu(
    resulting_type: ConversionType,
    on: Conversion,
    table: Conversion
) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L110"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `eint`

```python
eint(width: int) → ConversionType
```

Get encrypted unsigned integer type (e.g., !FHE.eint<3>, !FHE.eint<5>). 

---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L182"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `element_typeof`

```python
element_typeof(value: Union[Conversion, ConversionType]) → ConversionType
```

Get type corresponding to the elements of a tensor type. 

---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L2395"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `encrypt`

```python
encrypt(resulting_type: ConversionType, x: Conversion) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L2402"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `equal`

```python
equal(resulting_type: ConversionType, x: Conversion, y: Conversion) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L264"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `error`

```python
error(highlights: Mapping[Node, Union[str, list[str]]])
```

Fail compilation with an error. 



**Args:**
  highlights (Mapping[Node, Union[str, List[str]]]):  nodes to highlight along with messages 

---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L116"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `esint`

```python
esint(width: int) → ConversionType
```

Get encrypted signed integer type (e.g., !FHE.esint<3>, !FHE.esint<5>). 

---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L2423"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `extract_bits`

```python
extract_bits(
    resulting_type: ConversionType,
    x: Conversion,
    bits: Union[int, integer, slice]
) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L2547"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `flatten`

```python
flatten(x: Conversion) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L468"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `for_loop`

```python
for_loop(
    lower_bound: int,
    upper_bound: int,
    body: Union[Callable[[Conversion], Optional[Conversion]], Callable[[Conversion, Conversion], Optional[Conversion]]],
    output: Optional[Conversion] = None,
    step: int = 1
) → Optional[Conversion]
```

Create a for loop. 



**Args:**
  lower_bound (int):  starting position of the for loop 

 upper_bound (int):  upper bound of the for loop 

 body (Union[  Callable[[Conversion], Optional[Conversion]],  Callable[[Conversion, Conversion], Optional[Conversion]],  ]):  body of the for loop 

 output (Optional[Conversion], default = None):  initial value of the output of the for loop 

 step (int, default = 1):  step between the iterations of the for loop 



**Returns:**
  Optional[Conversion]:  None if output is None  conversion of the created operation otherwise 



**Notes:**

> - if output is None body builder must take a single indexing variable argument - if output is not None body builder must take an indexing variable and output arguments and body must end with an scf yield operation with the updated output 

---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L201"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fork_type`

```python
fork_type(
    type_: ConversionType,
    bit_width: Optional[int] = None,
    is_signed: Optional[bool] = None,
    shape: Optional[tuple[int, ]] = None
) → ConversionType
```

Fork a type with some properties update. 

---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L4020"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_partition_name`

```python
get_partition_name(partition: CryptoParams) → str
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L2550"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `greater`

```python
greater(
    resulting_type: ConversionType,
    x: Conversion,
    y: Conversion
) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L2553"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `greater_equal`

```python
greater_equal(
    resulting_type: ConversionType,
    x: Conversion,
    y: Conversion
) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L104"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `i`

```python
i(width: int) → ConversionType
```

Get clear signless integer type (e.g., i3, i5). 

---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L2561"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `identity`

```python
identity(
    resulting_type: ConversionType,
    x: Conversion,
    force_noise_refresh: bool
) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L2595"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `index`

```python
index(
    resulting_type: ConversionType,
    x: Conversion,
    index: Sequence[Union[int, integer, slice, ndarray, list, Conversion]]
) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L122"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `index_type`

```python
index_type() → ConversionType
```

Get index type. 

---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L275"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_bit_width_compatible`

```python
is_bit_width_compatible(*args: Optional[ConversionType, Conversion]) → bool
```

Check if conversion types are compatible in terms of bit-width. 

---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L2610"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `less`

```python
less(resulting_type: ConversionType, x: Conversion, y: Conversion) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L2613"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `less_equal`

```python
less_equal(
    resulting_type: ConversionType,
    x: Conversion,
    y: Conversion
) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L224"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `location`

```python
location() → Location
```

Create an MLIR location from the node that is being converted. 

---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L2621"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `lsb`

```python
lsb(resulting_type: ConversionType, x: Conversion) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L2628"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `matmul`

```python
matmul(
    resulting_type: ConversionType,
    x: Conversion,
    y: Conversion
) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L2706"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `maximum`

```python
maximum(
    resulting_type: ConversionType,
    x: Conversion,
    y: Conversion
) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L2779"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `maxpool2d`

```python
maxpool2d(
    resulting_type: ConversionType,
    x: Conversion,
    kernel_shape: tuple[int, ],
    strides: Sequence[int],
    dilations: Sequence[int]
)
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L2828"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `min_max`

```python
min_max(
    resulting_type: ConversionType,
    x: Conversion,
    axes: Union[int, integer, Sequence[Union[int, integer]]] = (),
    keep_dims: bool = False,
    operation: str
)
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L2847"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `minimum`

```python
minimum(
    resulting_type: ConversionType,
    x: Conversion,
    y: Conversion
) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L1467"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L1407"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `minimum_maximum_with_trick`

```python
minimum_maximum_with_trick(
    resulting_type: ConversionType,
    x: Conversion,
    y: Conversion,
    x_minus_y_dtype: Integer,
    intermediate_table: list[int]
) → Conversion
```

Calculate minimum or maximum between two encrypted values using minimum or maximum trick. 

Idea:  min(x, y) <==> min(x - y, 0) + y  max(x, y) <==> max(x - y, 0) + y 

Additional Args:  x_minus_y_dtype (Integer):  minimal dtype that can be used to store x - y without overflows 

---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L2921"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `mul`

```python
mul(resulting_type: ConversionType, x: Conversion, y: Conversion) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L3006"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L1708"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L3097"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `multivariate_multi_tlu`

```python
multivariate_multi_tlu(
    resulting_type: ConversionType,
    xs: list[Conversion],
    tables: Any,
    mapping: Any
)
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L3086"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `multivariate_tlu`

```python
multivariate_tlu(
    resulting_type: ConversionType,
    xs: list[Conversion],
    table: Sequence[int]
) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L3109"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `neg`

```python
neg(resulting_type: ConversionType, x: Conversion) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L128"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `none_type`

```python
none_type() → ConversionType
```

Get none type. 

---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L3127"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `not_equal`

```python
not_equal(
    resulting_type: ConversionType,
    x: Conversion,
    y: Conversion
) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L3135"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ones`

```python
ones(resulting_type: ConversionType) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L300"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `operation`

```python
operation(
    operation: Callable,
    resulting_type: Optional[ConversionType],
    *args,
    original_bit_width: Optional[int] = None,
    use_cache: bool = True,
    **kwargs
) → Conversion
```

Create a conversion from an MLIR operation. 



**Args:**
  operation (Callable):  MLIR operation to create (e.g., fhe.AddEintOp) 

 resulting_type (Optional[ConversionType]):  optional type of the output of the operation 

 *args (Any):  args to pass to the operation 

 original_bit_width (Optional[int], default = None):  original bit width of the resulting conversion 

 use_cache (bool, default = True):  whether to use the operation cache or not 

 *kwargs (Any):  kwargs to pass to the operation 

**Returns:**
  Conversion:  resulting conversion 

---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L1325"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `pack_multivariate_inputs`

```python
pack_multivariate_inputs(xs: list[Conversion]) → Conversion
```

Packs inputs of multivariate table lookups. 



**Args:**
  xs (List[Conversion]):  operands 



**Returns:**
  Conversion:  packed operands 

---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L3932"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `reinterpret`

```python
reinterpret(
    x: Conversion,
    bit_width: int,
    signed: Optional[bool] = None
) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L3149"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `relu`

```python
relu(resulting_type: ConversionType, x: Conversion) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L3205"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `reshape`

```python
reshape(x: Conversion, shape: tuple[int, ]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L3336"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `round_bit_pattern`

```python
round_bit_pattern(
    resulting_type: ConversionType,
    x: Conversion,
    lsbs_to_remove: int,
    exactness: Exactness,
    overflow_detected: bool
) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L2412"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `safe_reduce_precision`

```python
safe_reduce_precision(x: Conversion, bit_width: int) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L3472"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L2405"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `shift_left_at_constant_precision`

```python
shift_left_at_constant_precision(x: Conversion, rank: int) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L3673"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `sub`

```python
sub(resulting_type: ConversionType, x: Conversion, y: Conversion) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L3706"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L134"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `tensor`

```python
tensor(element_type: ConversionType, shape: tuple[int, ]) → ConversionType
```

Get tensor type (e.g., tensor<5xi3>, tensor<3x2x!FHE.eint<5>>). 

---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L3755"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `tensorize`

```python
tensorize(x: Conversion) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L3766"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `tlu`

```python
tlu(resulting_type: ConversionType, on: Conversion, table: Sequence[int])
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L3836"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_signed`

```python
to_signed(x: Conversion) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L3858"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_signedness`

```python
to_signedness(x: Conversion, of: ConversionType) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L3861"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_unsigned`

```python
to_unsigned(x: Conversion) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L3883"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `transpose`

```python
transpose(
    resulting_type: ConversionType,
    x: Conversion,
    axes: Sequence[int] = ()
)
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L3894"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `tree_add`

```python
tree_add(resulting_type: ConversionType, xs: list[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L3912"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `truncate_bit_pattern`

```python
truncate_bit_pattern(x: Conversion, lsbs_to_remove: int) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L742"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `try_comparison_with_clipping_trick`

```python
try_comparison_with_clipping_trick(
    resulting_type: ConversionType,
    x: Conversion,
    y: Conversion,
    accept: set[Comparison]
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

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L144"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `typeof`

```python
typeof(value: Optional[ValueDescription, Node]) → ConversionType
```

Get type corresponding to a value or a node. 

---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L3950"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../frontends/concrete-python/concrete/fhe/mlir/context.py#L4008"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `zeros`

```python
zeros(resulting_type: ConversionType) → Conversion
```






