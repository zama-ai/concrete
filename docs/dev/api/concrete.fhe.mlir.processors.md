<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/processors/__init__.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.mlir.processors`
Declaration of `GraphProcessor` class. 

**Global Variables**
---------------
- **assign_bit_widths**
- **check_integer_only**
- **process_rounding**
- **all**


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/processors/__init__.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `GraphProcessor`
GraphProcessor base class, to define the API for a graph processing pipeline. 




---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/processors/__init__.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `apply`

```python
apply(graph: Graph)
```

Process the graph. 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/mlir/processors/__init__.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `error`

```python
error(graph: Graph, highlights: Mapping[Node, Union[str, List[str]]])
```

Fail processing with an error. 



**Args:**
  graph (Graph):  graph being processed 

 highlights (Mapping[Node, Union[str, List[str]]]):  nodes to highlight along with messages 


