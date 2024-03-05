<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/processors/assign_bit_widths.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.mlir.processors.assign_bit_widths`
Declaration of `AssignBitWidths` graph processor. 



---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/processors/assign_bit_widths.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AssignBitWidths`
AssignBitWidths graph processor, to assign proper bit-widths to be compatible with FHE. 

There are two modes: 
- Single Precision, where all encrypted values have the same precision. 
- Multi Precision, where encrypted values can have different precisions. 

There is preference list for comparison strategies. 
- Strategies will be traversed in order and bit-widths  will be assigned according to the first available strategy. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/processors/assign_bit_widths.py#L42"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    single_precision: bool,
    composable: bool,
    comparison_strategy_preference: List[ComparisonStrategy],
    bitwise_strategy_preference: List[BitwiseStrategy],
    shifts_with_promotion: bool,
    multivariate_strategy_preference: List[MultivariateStrategy],
    min_max_strategy_preference: List[MinMaxStrategy]
)
```








---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/processors/assign_bit_widths.py#L60"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `apply`

```python
apply(graph: Graph)
```






---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/processors/assign_bit_widths.py#L126"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AdditionalConstraints`
AdditionalConstraints class to customize bit-width assignment step easily. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/processors/assign_bit_widths.py#L146"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    optimizer: Optimize,
    graph: Graph,
    bit_widths: Dict[Node, Int],
    comparison_strategy_preference: List[ComparisonStrategy],
    bitwise_strategy_preference: List[BitwiseStrategy],
    shifts_with_promotion: bool,
    multivariate_strategy_preference: List[MultivariateStrategy],
    min_max_strategy_preference: List[MinMaxStrategy]
)
```








---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/processors/assign_bit_widths.py#L227"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `all_inputs_are_encrypted`

```python
all_inputs_are_encrypted(node: Node, preds: List[Node]) → bool
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/processors/assign_bit_widths.py#L285"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `bitwise`

```python
bitwise(node: Node, preds: List[Node])
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/processors/assign_bit_widths.py#L258"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `comparison`

```python
comparison(node: Node, preds: List[Node])
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/processors/assign_bit_widths.py#L219"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `constraint`

```python
constraint(node: Node, constraint: BoolRef)
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/processors/assign_bit_widths.py#L167"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `generate_for`

```python
generate_for(node: Node, bit_width: <function Int at ADDRESS>)
```

Generate additional constraints for a node. 



**Args:**
  node (Node):  node to generate constraints for 

 bit_width (z3.Int):  symbolic bit-width which will be assigned to node once constraints are solved 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/processors/assign_bit_widths.py#L233"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `has_overflow_protection`

```python
has_overflow_protection(node: Node, preds: List[Node]) → bool
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/processors/assign_bit_widths.py#L244"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `inputs_and_output_share_precision`

```python
inputs_and_output_share_precision(node: Node, preds: List[Node])
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/processors/assign_bit_widths.py#L249"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `inputs_require_one_more_bit`

```python
inputs_require_one_more_bit(node: Node, preds: List[Node])
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/processors/assign_bit_widths.py#L240"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `inputs_share_precision`

```python
inputs_share_precision(node: Node, preds: List[Node])
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/processors/assign_bit_widths.py#L345"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `min_max`

```python
min_max(node: Node, preds: List[Node])
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/processors/assign_bit_widths.py#L318"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `multivariate`

```python
multivariate(node: Node, preds: List[Node])
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/processors/assign_bit_widths.py#L230"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `some_inputs_are_clear`

```python
some_inputs_are_clear(node: Node, preds: List[Node]) → bool
```






