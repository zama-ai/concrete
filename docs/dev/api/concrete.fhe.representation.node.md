<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/representation/node.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.representation.node`
Declaration of `Node` class. 

**Global Variables**
---------------
- **KWARGS_IGNORED_IN_FORMATTING**
- **NODES_THAT_HAVE_TLU_WHEN_ALL_INPUTS_ARE_ENCRYPTED**


---

<a href="../../frontends/concrete-python/concrete/fhe/representation/node.py#L26"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Node`
Node class, to represent computation in a computation graph. 

<a href="../../frontends/concrete-python/concrete/fhe/representation/node.py#L151"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    inputs: list[ValueDescription],
    output: ValueDescription,
    operation: Operation,
    evaluator: Callable,
    properties: Optional[dict[str, Any]] = None
)
```






---

#### <kbd>property</kbd> conversion_have_table_lookup

Get whether the node will have table lookups during execution. 



**Returns:**
  bool:  True if the node will have table lookups during execution, False otherwise 

---

#### <kbd>property</kbd> converted_to_table_lookup

Get whether the node is converted to a table lookup during MLIR conversion. 



**Returns:**
  bool:  True if the node is converted to a table lookup, False otherwise 

---

#### <kbd>property</kbd> is_fusable

Get whether the node is can be fused into a table lookup. 



**Returns:**
  bool:  True if the node can be fused into a table lookup, False otherwise 



---

<a href="../../frontends/concrete-python/concrete/fhe/representation/node.py#L46"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `constant`

```python
constant(constant: Any) → Node
```

Create an Operation.Constant node. 



**Args:**
  constant (Any):  constant to represent 



**Returns:**
  Node:  node representing constant 



**Raises:**
  ValueError:  if the constant is not representable 

---

<a href="../../frontends/concrete-python/concrete/fhe/representation/node.py#L280"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `format`

```python
format(predecessors: list[str], maximum_constant_length: int = 45) → str
```

Get the textual representation of the `Node` (dependent to preds). 



**Args:**
  predecessors (List[str]):  predecessor names to this node 

 maximum_constant_length (int, default = 45):  maximum length of formatted constants 



**Returns:**
  str:  textual representation of the `Node` (dependent to preds) 

---

<a href="../../frontends/concrete-python/concrete/fhe/representation/node.py#L73"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `generic`

```python
generic(
    name: str,
    inputs: list[ValueDescription],
    output: ValueDescription,
    operation: Callable,
    args: Optional[tuple[Any, ]] = None,
    kwargs: Optional[dict[str, Any]] = None,
    attributes: Optional[dict[str, Any]] = None
)
```

Create an Operation.Generic node. 



**Args:**
  name (str):  name of the operation 

 inputs (List[ValueDescription]):  inputs to the operation 

 output (ValueDescription):  output of the operation 

 operation (Callable):  operation itself 

 args (Optional[Tuple[Any, ...]]):  args to pass to operation during evaluation 

 kwargs (Optional[Dict[str, Any]]):  kwargs to pass to operation during evaluation 

 attributes (Optional[Dict[str, Any]]):  attributes of the operation 



**Returns:**
  Node:  node representing operation 

---

<a href="../../frontends/concrete-python/concrete/fhe/representation/node.py#L132"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `input`

```python
input(name: str, value: ValueDescription) → Node
```

Create an Operation.Input node. 



**Args:**
  name (Any):  name of the input 

 value (Any):  value of the input 



**Returns:**
  Node:  node representing input 

---

<a href="../../frontends/concrete-python/concrete/fhe/representation/node.py#L385"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `label`

```python
label() → str
```

Get the textual representation of the `Node` (independent of preds). 



**Returns:**
  str:  textual representation of the `Node` (independent of preds). 


