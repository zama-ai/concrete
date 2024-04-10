<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.mlir.converter`
Declaration of `Converter` class. 

**Global Variables**
---------------
- **MAXIMUM_TLU_BIT_WIDTH**


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L31"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Converter`
Converter class, to convert a computation graph to MLIR. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L38"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(configuration: Configuration)
```








---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L282"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add`

```python
add(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L286"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `array`

```python
array(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L290"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `assign_static`

```python
assign_static(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L299"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `bitwise_and`

```python
bitwise_and(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L307"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `bitwise_or`

```python
bitwise_or(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L315"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `bitwise_xor`

```python
bitwise_xor(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L323"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `broadcast_to`

```python
broadcast_to(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L327"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `concatenate`

```python
concatenate(ctx: Context, node: Node, preds: List[Conversion])
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L334"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `constant`

```python
constant(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L338"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `conv1d`

```python
conv1d(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L342"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `conv2d`

```python
conv2d(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L355"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `conv3d`

```python
conv3d(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L105"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `convert`

```python
convert(
    graph: Graph,
    mlir_context: <locals>Context,
    name: str = 'main'
) → Module
```

Convert a computation graph to MLIR. 



**Args:**
  graph (Graph):  graph to convert 

 mlir_context (MlirContext):  MLIR Context to use for module generation 

 name (str):  name of the function to convert 

Return:  MlirModule:  In-memory MLIR module corresponding to the graph 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L41"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `convert_many`

```python
convert_many(graphs: Dict[str, Graph], mlir_context: <locals>Context) → Module
```

Convert multiple computation graphs to an MLIR module. 



**Args:**
  graphs (Dict[str, Graph]):  graphs to convert 

 mlir_context (MlirContext):  MLIR Context to use for module generation 

Return:  MlirModule:  In-memory MLIR module corresponding to the graph 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L359"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `copy`

```python
copy(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L363"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dot`

```python
dot(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L367"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dynamic_tlu`

```python
dynamic_tlu(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L371"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `equal`

```python
equal(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L379"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `expand_dims`

```python
expand_dims(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L383"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `extract_bit_pattern`

```python
extract_bit_pattern(
    ctx: Context,
    node: Node,
    preds: List[Conversion]
) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L387"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `greater`

```python
greater(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L395"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `greater_equal`

```python
greater_equal(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L403"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `identity`

```python
identity(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L407"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `index_static`

```python
index_static(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L415"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `left_shift`

```python
left_shift(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L429"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `less`

```python
less(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L437"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `less_equal`

```python
less_equal(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L445"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `matmul`

```python
matmul(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L449"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `maximum`

```python
maximum(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L457"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `maxpool1d`

```python
maxpool1d(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L461"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `maxpool2d`

```python
maxpool2d(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L471"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `maxpool3d`

```python
maxpool3d(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L475"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `minimum`

```python
minimum(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L483"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `multiply`

```python
multiply(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L487"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `negative`

```python
negative(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L245"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `node`

```python
node(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```

Convert a computation graph node into MLIR. 



**Args:**
  ctx (Context):  conversion context 

 node (Node):  node to convert 

 preds (List[Conversion]):  conversions of ordered predecessors of the node 

Return:  Conversion:  conversion object corresponding to node 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L491"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `not_equal`

```python
not_equal(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L499"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ones`

```python
ones(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L206"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `process`

```python
process(graphs: Dict[str, Graph])
```

Process a computation graph for MLIR conversion. 



**Args:**
  graphs (Dict[str, Graph]):  graphs to process 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L503"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `relu`

```python
relu(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L507"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `reshape`

```python
reshape(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L511"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `right_shift`

```python
right_shift(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L525"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `round_bit_pattern`

```python
round_bit_pattern(
    ctx: Context,
    node: Node,
    preds: List[Conversion]
) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L136"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `simplify_tag`

```python
simplify_tag(configuration: Configuration, tag: str) → str
```

Keep only `n` higher tag parts. 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L568"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `squeeze`

```python
squeeze(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L131"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `stdout_with_ansi_support`

```python
stdout_with_ansi_support() → bool
```

Detect if ansi characters can be used (e.g. not the case in notebooks). 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L555"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `subtract`

```python
subtract(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L559"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `sum`

```python
sum(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L639"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `tlu`

```python
tlu(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L584"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `tlu_adjust`

```python
tlu_adjust(table, variable_input, target_bit_width, clipping, reduce_precision)
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L148"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `trace_progress`

```python
trace_progress(
    configuration: Configuration,
    progress_index: int,
    nodes: List[Node]
)
```

Add a trace_message for progress. 



**Args:**
  configuration:  configuration for title, tags options 

 progress_index:  index of the next node to process 

 nodes:  all nodes 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L797"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `transpose`

```python
transpose(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L805"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `truncate_bit_pattern`

```python
truncate_bit_pattern(
    ctx: Context,
    node: Node,
    preds: List[Conversion]
) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L809"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `where`

```python
where(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/converter.py#L813"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `zeros`

```python
zeros(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```






