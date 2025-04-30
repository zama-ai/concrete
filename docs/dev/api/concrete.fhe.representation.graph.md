<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/representation/graph.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.representation.graph`
Declaration of `Graph` class. 

**Global Variables**
---------------
- **P_ERROR_PER_ERROR_SIZE_CACHE**


---

<a href="../../frontends/concrete-python/concrete/fhe/representation/graph.py#L27"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Graph`
Graph class, to represent computation graphs. 

<a href="../../frontends/concrete-python/concrete/fhe/representation/graph.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    graph: MultiDiGraph,
    input_nodes: dict[int, Node],
    output_nodes: dict[int, Node],
    name: str,
    is_direct: bool = False,
    location: str = ''
)
```






---

#### <kbd>property</kbd> inputs_count

Returns the number of inputs of the graph. 

---

#### <kbd>property</kbd> outputs_count

Returns the number of outputs of the graph. 



---

<a href="../../frontends/concrete-python/concrete/fhe/representation/graph.py#L211"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `draw`

```python
draw(
    horizontal: bool = False,
    save_to: Optional[Path, str] = None,
    show: bool = False
) → Path
```

Draw the graph. 

That this function requires the python `pygraphviz` package which itself requires the installation of `graphviz` packages 

(see https://pygraphviz.github.io/documentation/stable/install.html) 



**Args:**
  horizontal (bool, default = False):  whether to draw horizontally 

 save_to (Optional[Path], default = None):  path to save the drawing  a temporary file will be used if it's None 

 show (bool, default = False):  whether to show the drawing using matplotlib 



**Returns:**
  Path:  path to the drawing 

---

<a href="../../frontends/concrete-python/concrete/fhe/representation/graph.py#L89"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate`

```python
evaluate(
    *args: Any,
    p_error: Optional[float] = None
) → dict[Node, Union[bool_, integer, floating, ndarray]]
```

Perform the computation `Graph` represents and get resulting values for all nodes. 



**Args:**
  *args (List[Any]):  inputs to the computation 

 p_error (Optional[float]):  probability of error for table lookups 



**Returns:**
  Dict[Node, Union[np.bool\_, np.integer, np.floating, np.ndarray]]:  nodes and their values during computation 

---

<a href="../../frontends/concrete-python/concrete/fhe/representation/graph.py#L350"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `format`

```python
format(
    maximum_constant_length: int = 25,
    highlighted_nodes: Optional[dict[Node, list[str]]] = None,
    highlighted_result: Optional[list[str]] = None,
    show_types: bool = True,
    show_bounds: bool = True,
    show_tags: bool = True,
    show_locations: bool = False,
    show_assigned_bit_widths: bool = False
) → str
```

Get the textual representation of the `Graph`. 



**Args:**
  maximum_constant_length (int, default = 25):  maximum length of formatted constants 

 highlighted_nodes (Optional[Dict[Node, List[str]]], default = None):  nodes to be highlighted and their corresponding messages 

 highlighted_result (Optional[List[str]], default = None):  messages corresponding to highlighted return line 

 show_types (bool, default = True):  whether to show types of nodes 

 show_bounds (bool, default = True):  whether to show bounds of nodes 

 show_tags (bool, default = True):  whether to show tags of nodes 

 show_locations (bool, default = False):  whether to show line information of nodes 

 show_assigned_bit_widths (bool, default = False)  whether to show assigned bit width of nodes instead of their original bit width 



**Returns:**
  str:  textual representation of the `Graph` 

---

<a href="../../frontends/concrete-python/concrete/fhe/representation/graph.py#L589"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `format_bit_width_assignments`

```python
format_bit_width_assignments() → str
```

Get the textual representation of bit width assignments of the graph. 



**Returns:**
  str:  textual representation of bit width assignments of the graph 

---

<a href="../../frontends/concrete-python/concrete/fhe/representation/graph.py#L572"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `format_bit_width_constraints`

```python
format_bit_width_constraints() → str
```

Get the textual representation of bit width constraints of the graph. 



**Returns:**
  str:  textual representation of bit width constraints of the graph 

---

<a href="../../frontends/concrete-python/concrete/fhe/representation/graph.py#L935"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `integer_range`

```python
integer_range(
    tag_filter: Optional[str, list[str], Pattern] = None,
    operation_filter: Optional[str, list[str], Pattern] = None,
    is_encrypted_filter: Optional[bool] = None,
    custom_filter: Optional[Callable[[Node], bool]] = None
) → Optional[tuple[int, int]]
```

Get integer range of the graph. 

Only nodes after filtering will be used to calculate the result. 



**Args:**
  tag_filter (Optional[Union[str, List[str], re.Pattern]], default = None):  filter for tags 

 operation_filter (Optional[Union[str, List[str], re.Pattern]], default = None):  filter for operations 

 is_encrypted_filter (Optional[bool], default = None)  filter for encryption status 

 custom_filter (Optional[Callable[[Node], bool]], default = None):  flexible filter 



**Returns:**
  Optional[Tuple[int, int]]:  minimum and maximum integer value observed during inputset evaluation  if there are no integer nodes matching the query, result is None 

---

<a href="../../frontends/concrete-python/concrete/fhe/representation/graph.py#L885"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `maximum_integer_bit_width`

```python
maximum_integer_bit_width(
    tag_filter: Optional[str, list[str], Pattern] = None,
    operation_filter: Optional[str, list[str], Pattern] = None,
    is_encrypted_filter: Optional[bool] = None,
    custom_filter: Optional[Callable[[Node], bool]] = None,
    assigned_bit_width: bool = False
) → int
```

Get maximum integer bit-width within the graph. 

Only nodes after filtering will be used to calculate the result. 



**Args:**
  tag_filter (Optional[Union[str, List[str], re.Pattern]], default = None):  filter for tags 

 operation_filter (Optional[Union[str, List[str], re.Pattern]], default = None):  filter for operations 

 is_encrypted_filter (Optional[bool], default = None):  filter for encryption status 

 custom_filter (Optional[Callable[[Node], bool]], default = None):  flexible filter 

 assigned_bit_width (Optional[bool], default = None):  whether to query on assigned bit-widths 



**Returns:**
  int:  maximum integer bit-width within the graph  if there are no integer nodes matching the query, result is -1 

---

<a href="../../frontends/concrete-python/concrete/fhe/representation/graph.py#L629"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `measure_bounds`

```python
measure_bounds(
    inputset: Union[Iterable[Any], Iterable[tuple[Any, ]]]
) → dict[Node, dict[str, Union[integer, floating]]]
```

Evaluate the `Graph` using an inputset and measure bounds. 

inputset is either an iterable of anything for a single parameter 

or 

an iterable of tuples of anything (of rank number of parameters) for multiple parameters 

e.g., 

.. code-block:: python 

 inputset = [1, 3, 5, 2, 4]  def f(x):  ... 

 inputset = [(1, 2), (2, 4), (3, 1), (2, 2)]  def g(x, y):  ... 



**Args:**
  inputset (Union[Iterable[Any], Iterable[Tuple[Any, ...]]]):  inputset to use 



**Returns:**
  Dict[Node, Dict[str, Union[np.integer, np.floating]]]:  bounds of each node in the `Graph` 

---

<a href="../../frontends/concrete-python/concrete/fhe/representation/graph.py#L749"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ordered_inputs`

```python
ordered_inputs() → list[Node]
```

Get the input nodes of the `Graph`, ordered by their indices. 



**Returns:**
  List[Node]:  ordered input nodes 

---

<a href="../../frontends/concrete-python/concrete/fhe/representation/graph.py#L760"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ordered_outputs`

```python
ordered_outputs() → list[Node]
```

Get the output nodes of the `Graph`, ordered by their indices. 



**Returns:**
  List[Node]:  ordered output nodes 

---

<a href="../../frontends/concrete-python/concrete/fhe/representation/graph.py#L771"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ordered_preds_of`

```python
ordered_preds_of(node: Node) → list[Node]
```

Get predecessors of `node`, ordered by their indices. 



**Args:**
  node (Node):  node whose predecessors are requested 



**Returns:**
  List[Node]:  ordered predecessors of `node`. 

---

<a href="../../frontends/concrete-python/concrete/fhe/representation/graph.py#L790"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `prune_useless_nodes`

```python
prune_useless_nodes()
```

Remove unreachable nodes from the graph. 

---

<a href="../../frontends/concrete-python/concrete/fhe/representation/graph.py#L806"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `query_nodes`

```python
query_nodes(
    tag_filter: Optional[str, list[str], Pattern] = None,
    operation_filter: Optional[str, list[str], Pattern] = None,
    is_encrypted_filter: Optional[bool] = None,
    custom_filter: Optional[Callable[[Node], bool]] = None,
    ordered: bool = False
) → list[Node]
```

Query nodes within the graph. 

Filters work like so:  str -> nodes without exact match is skipped  List[str] -> nodes without exact match with one of the strings in the list is skipped  re.Pattern -> nodes without pattern match is skipped 



**Args:**
  tag_filter (Optional[Union[str, List[str], re.Pattern]], default = None):  filter for tags 

 operation_filter (Optional[Union[str, List[str], re.Pattern]], default = None):  filter for operations 

 is_encrypted_filter (Optional[bool], default = None)  filter for encryption status 

 custom_filter (Optional[Callable[[Node], bool]], default = None):  flexible filter 

 ordered (bool)  whether to apply topological sorting before filtering nodes 



**Returns:**
  List[Node]:  filtered nodes 

---

<a href="../../frontends/concrete-python/concrete/fhe/representation/graph.py#L700"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update_with_bounds`

```python
update_with_bounds(bounds: dict[Node, dict[str, Union[integer, floating]]])
```

Update `ValueDescription`s within the `Graph` according to measured bounds. 



**Args:**
  bounds (Dict[Node, Dict[str, Union[np.integer, np.floating]]]):  bounds of each node in the `Graph` 


---

<a href="../../frontends/concrete-python/concrete/fhe/representation/graph.py#L1006"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `GraphProcessor`
GraphProcessor base class, to define the API for a graph processing pipeline. 

Process a single graph. 




---

<a href="../../frontends/concrete-python/concrete/fhe/representation/graph.py#L1013"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `apply`

```python
apply(graph: Graph)
```

Process the graph. 

---

<a href="../../frontends/concrete-python/concrete/fhe/representation/graph.py#L1019"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `error`

```python
error(graph: Graph, highlights: Mapping[Node, Union[str, list[str]]])
```

Fail processing with an error. 



**Args:**
  graph (Graph):  graph being processed 

 highlights (Mapping[Node, Union[str, List[str]]]):  nodes to highlight along with messages 


---

<a href="../../frontends/concrete-python/concrete/fhe/representation/graph.py#L1044"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MultiGraphProcessor`
MultiGraphProcessor base class, to define the API for a multiple graph processing pipeline. 

Processes multiple graphs at once. 




---

<a href="../../frontends/concrete-python/concrete/fhe/representation/graph.py#L1057"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `apply`

```python
apply(graph: Graph)
```

Process a single graph. 

---

<a href="../../frontends/concrete-python/concrete/fhe/representation/graph.py#L1051"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `apply_many`

```python
apply_many(graphs: dict[str, Graph])
```

Process a dictionary of graphs. 

---

<a href="../../frontends/concrete-python/concrete/fhe/representation/graph.py#L1019"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `error`

```python
error(graph: Graph, highlights: Mapping[Node, Union[str, list[str]]])
```

Fail processing with an error. 



**Args:**
  graph (Graph):  graph being processed 

 highlights (Mapping[Node, Union[str, List[str]]]):  nodes to highlight along with messages 


