<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/mlir/processors/assign_bit_widths.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.mlir.processors.assign_bit_widths`
Declaration of `AssignBitWidths` graph processor. 



---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/processors/assign_bit_widths.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AssignBitWidths`
AssignBitWidths graph processor, to assign proper bit-widths to be compatible with FHE. 

There are two modes: 
- Single Precision, where all encrypted values have the same precision. 
- Multi Precision, where encrypted values can have different precisions. 

There is preference list for comparison strategies. 
- Strategies will be traversed in order and bit-widths  will be assigned according to the first available strategy. 

<a href="../../frontends/concrete-python/concrete/fhe/mlir/processors/assign_bit_widths.py#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    single_precision: bool,
    composition_rules: list[CompositionRule],
    comparison_strategy_preference: list[ComparisonStrategy],
    bitwise_strategy_preference: list[BitwiseStrategy],
    shifts_with_promotion: bool,
    multivariate_strategy_preference: list[MultivariateStrategy],
    min_max_strategy_preference: list[MinMaxStrategy]
)
```








---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/processors/assign_bit_widths.py#L57"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `apply_many`

```python
apply_many(graphs: dict[str, Graph])
```






---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/processors/assign_bit_widths.py#L129"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AdditionalConstraints`
AdditionalConstraints class to customize bit-width assignment step easily. 

<a href="../../frontends/concrete-python/concrete/fhe/mlir/processors/assign_bit_widths.py#L149"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    optimizer: Optimize,
    graph: Graph,
    bit_widths: dict[Node, Int],
    comparison_strategy_preference: list[ComparisonStrategy],
    bitwise_strategy_preference: list[BitwiseStrategy],
    shifts_with_promotion: bool,
    multivariate_strategy_preference: list[MultivariateStrategy],
    min_max_strategy_preference: list[MinMaxStrategy]
)
```








---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/processors/assign_bit_widths.py#L234"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `all_inputs_are_encrypted`

```python
all_inputs_are_encrypted(node: Node, preds: list[Node]) → bool
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/processors/assign_bit_widths.py#L292"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `bitwise`

```python
bitwise(node: Node, preds: list[Node])
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/processors/assign_bit_widths.py#L265"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `comparison`

```python
comparison(node: Node, preds: list[Node])
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/processors/assign_bit_widths.py#L226"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `constraint`

```python
constraint(node: Node, constraint: BoolRef)
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/processors/assign_bit_widths.py#L170"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `generate_for`

```python
generate_for(node: Node, bit_width: <function Int at ADDRESS>)
```

Generate additional constraints for a node. 



**Args:**
  node (Node):  node to generate constraints for 

 bit_width (z3.Int):  symbolic bit-width which will be assigned to node once constraints are solved 

---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/processors/assign_bit_widths.py#L240"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `has_overflow_protection`

```python
has_overflow_protection(node: Node, preds: list[Node]) → bool
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/processors/assign_bit_widths.py#L251"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `inputs_and_output_share_precision`

```python
inputs_and_output_share_precision(node: Node, preds: list[Node])
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/processors/assign_bit_widths.py#L256"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `inputs_require_one_more_bit`

```python
inputs_require_one_more_bit(node: Node, preds: list[Node])
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/processors/assign_bit_widths.py#L247"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `inputs_share_precision`

```python
inputs_share_precision(node: Node, preds: list[Node])
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/processors/assign_bit_widths.py#L390"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `min_max`

```python
min_max(node: Node, preds: list[Node])
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/processors/assign_bit_widths.py#L352"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `minimum_maximum`

```python
minimum_maximum(node: Node, preds: list[Node])
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/processors/assign_bit_widths.py#L325"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `multivariate`

```python
multivariate(node: Node, preds: list[Node])
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/processors/assign_bit_widths.py#L237"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `some_inputs_are_clear`

```python
some_inputs_are_clear(node: Node, preds: list[Node]) → bool
```






