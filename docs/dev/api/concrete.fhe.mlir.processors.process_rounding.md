<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/mlir/processors/process_rounding.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.mlir.processors.process_rounding`
Declaration of `ProcessRounding` graph processor. 



---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/processors/process_rounding.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ProcessRounding`
ProcessRounding graph processor, to analyze rounding and support regular operations on it. 

<a href="../../frontends/concrete-python/concrete/fhe/mlir/processors/process_rounding.py#L24"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(rounding_exactness: Exactness)
```








---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/processors/process_rounding.py#L30"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `apply`

```python
apply(graph: Graph)
```





---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/processors/process_rounding.py#L49"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `process_predecessors`

```python
process_predecessors(graph: Graph, node: Node)
```

Process predecessors of the rounding. 

---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/processors/process_rounding.py#L157"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `process_successors`

```python
process_successors(graph: Graph, node: Node)
```

Process successors of the rounding. 

---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/processors/process_rounding.py#L87"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `replace_with_tlu`

```python
replace_with_tlu(graph: Graph, node: Node)
```

Replace rounding node with a TLU node that simulates rounding. 

This is a special case where: 
- user wanted to remove 1-bits 
- there was an overflow 
- overflow protection was on 

Let's say user wanted to remove 1-bit from 3-bits, but there was an overflow: 
- [0, 1, 2, 3, 4, 5, 6, 7] would be mapped to [0, 2, 2, 4, 4, 6, 6, 8] 
- or in the actual implementation [(0)00, (0)01, (0)01, (0)10, (0)10, (0)11, (0)11, (1)00] 
- (first bit is the padding bit, which is overwritten on overflows) 
- so the input is 3-bits and the output needs to be 3-bits to store the result 
- which can't be achieved with rounding 
- so we just replace the rounding with a TLU 
- using the table [0, 2, 2, 4, 4, 6, 6, 8] 


