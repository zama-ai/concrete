<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.mlir.converter`
Declaration of `Converter` class. 

**Global Variables**
---------------
- **MAXIMUM_TLU_BIT_WIDTH**


---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L34"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Converter`
Converter class, to convert a computation graph to MLIR. 

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L42"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    configuration: Configuration,
    composition_rules: Optional[Iterable[CompositionRule]] = None
)
```








---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L322"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add`

```python
add(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L326"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `amax`

```python
amax(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L329"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `amin`

```python
amin(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L332"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `array`

```python
array(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L336"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `assign_dynamic`

```python
assign_dynamic(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L357"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `assign_static`

```python
assign_static(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L366"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `bitwise_and`

```python
bitwise_and(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L374"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `bitwise_or`

```python
bitwise_or(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L382"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `bitwise_xor`

```python
bitwise_xor(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L390"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `broadcast_to`

```python
broadcast_to(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L394"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `concatenate`

```python
concatenate(ctx: Context, node: Node, preds: List[Conversion])
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L401"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `constant`

```python
constant(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L405"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `conv1d`

```python
conv1d(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L409"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `conv2d`

```python
conv2d(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L422"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `conv3d`

```python
conv3d(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L144"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `convert`

```python
convert(graph: Graph, mlir_context: <locals>Context) → Module
```

Convert a computation graph to MLIR. 



**Args:**
  graph (Graph):  graph to convert 

 mlir_context (MlirContext):  MLIR Context to use for module generation 



Return:  MlirModule:  In-memory MLIR module corresponding to the graph 

---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L50"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L426"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `copy`

```python
copy(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L430"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dot`

```python
dot(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L434"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dynamic_tlu`

```python
dynamic_tlu(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L438"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `equal`

```python
equal(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L446"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `expand_dims`

```python
expand_dims(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L450"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `extract_bit_pattern`

```python
extract_bit_pattern(
    ctx: Context,
    node: Node,
    preds: List[Conversion]
) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L454"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `greater`

```python
greater(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L462"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `greater_equal`

```python
greater_equal(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L470"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `identity`

```python
identity(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L475"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `index_dynamic`

```python
index_dynamic(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L494"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `index_static`

```python
index_static(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L502"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `left_shift`

```python
left_shift(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L516"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `less`

```python
less(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L524"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `less_equal`

```python
less_equal(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L532"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `matmul`

```python
matmul(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L536"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `max`

```python
max(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L546"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `maximum`

```python
maximum(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L554"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `maxpool1d`

```python
maxpool1d(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L558"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `maxpool2d`

```python
maxpool2d(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L568"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `maxpool3d`

```python
maxpool3d(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L572"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `min`

```python
min(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L582"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `minimum`

```python
minimum(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L590"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `multiply`

```python
multiply(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L594"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `negative`

```python
negative(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L285"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L598"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `not_equal`

```python
not_equal(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L606"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ones`

```python
ones(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L242"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `process`

```python
process(graphs: Dict[str, Graph])
```

Process a computation graph for MLIR conversion. 



**Args:**
  graphs (Dict[str, Graph]):  graphs to process 

---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L610"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `relu`

```python
relu(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L614"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `reshape`

```python
reshape(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L618"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `right_shift`

```python
right_shift(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L632"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `round_bit_pattern`

```python
round_bit_pattern(
    ctx: Context,
    node: Node,
    preds: List[Conversion]
) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L172"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `simplify_tag`

```python
simplify_tag(configuration: Configuration, tag: str) → str
```

Keep only `n` higher tag parts. 

---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L679"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `squeeze`

```python
squeeze(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L167"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `stdout_with_ansi_support`

```python
stdout_with_ansi_support() → bool
```

Detect if ansi characters can be used (e.g. not the case in notebooks). 

---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L666"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `subtract`

```python
subtract(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L670"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `sum`

```python
sum(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L966"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `tfhers_from_native`

```python
tfhers_from_native(
    ctx: Context,
    node: Node,
    preds: List[Conversion]
) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L933"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `tfhers_to_native`

```python
tfhers_to_native(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L750"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `tlu`

```python
tlu(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L695"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `tlu_adjust`

```python
tlu_adjust(table, variable_input, target_bit_width, clipping, reduce_precision)
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L184"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L913"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `transpose`

```python
transpose(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L921"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `truncate_bit_pattern`

```python
truncate_bit_pattern(
    ctx: Context,
    node: Node,
    preds: List[Conversion]
) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L925"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `where`

```python
where(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/converter.py#L929"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `zeros`

```python
zeros(ctx: Context, node: Node, preds: List[Conversion]) → Conversion
```






