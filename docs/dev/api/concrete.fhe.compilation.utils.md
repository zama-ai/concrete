<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/utils.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.compilation.utils`
Declaration of various functions and constants related to compilation. 


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/utils.py#L23"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `validate_input_args`

```python
validate_input_args(
    client_specs: ClientSpecs,
    *args: Optional[int, ndarray, List]
) → List[Union[int, ndarray, NoneType]]
```

Validate input arguments. 



**Args:**
  client_specs (ClientSpecs):  client specification  *args (Optional[Union[int, np.ndarray, List]]):  argument(s) for evaluation 



**Returns:**
 
 - <b>`List[Optional[Union[int, np.ndarray]]]`</b>:  ordered validated args 


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/utils.py#L108"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fuse`

```python
fuse(graph: Graph, artifacts: Optional[DebugArtifacts] = None)
```

Fuse appropriate subgraphs in a graph to a single Operation.Generic node. 



**Args:**
  graph (Graph):  graph to search and update 

 artifacts (Optional[DebugArtifacts], default = None):  compilation artifacts to store information about the fusing process 



**Raises:**
  RuntimeError:  if there is a subgraph which needs to be fused cannot be fused 


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/utils.py#L188"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `find_float_subgraph_with_unique_terminal_node`

```python
find_float_subgraph_with_unique_terminal_node(
    graph: Graph,
    processed_terminal_nodes: Set[Node]
) → Optional[Tuple[Dict[Node, NoneType], Dict[Node, NoneType], Node]]
```

Find a subgraph with float computations that end with an integer output. 



**Args:**
  graph (Graph):  graph to search 

 processed_terminal_nodes (Set[Node]):  set of terminal nodes which have already been searched for float subgraphs 



**Returns:**
  Optional[Tuple[Dict[Node, None], Dict[Node, None], Node]]:  None if there are no such subgraphs,  tuple containing all nodes in the subgraph, start nodes of the subgraph,  and terminal node of the subgraph otherwise 


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/utils.py#L264"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `find_tlu_subgraph_with_multiple_variable_inputs_that_has_a_single_common_ancestor`

```python
find_tlu_subgraph_with_multiple_variable_inputs_that_has_a_single_common_ancestor(
    graph: Graph,
    processed_terminal_nodes: Set[Node]
) → Optional[Tuple[Dict[Node, NoneType], Dict[Node, NoneType], Node]]
```

Find a subgraph with a tlu computation that has multiple variable inputs     where all variable inputs share a common ancestor. 



**Args:**
  graph (Graph):  graph to search 

 processed_terminal_nodes (Set[Node]):  set of terminal nodes which have already been searched for tlu subgraphs 



**Returns:**
  Optional[Tuple[Dict[Node, None], Dict[Node, None], Node]]:  None if there are no such subgraphs,  tuple containing all nodes in the subgraph, start nodes of the subgraph,  and terminal node of the subgraph otherwise 


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/utils.py#L343"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `find_single_lca`

```python
find_single_lca(graph: Graph, nodes: List[Node]) → Optional[Node]
```

Find the single lowest common ancestor of a list of nodes. 



**Args:**
  graph (Graph):  graph to search for single lca 

 nodes (List[Node]):  nodes to find the single lca of 

Returns  Optional[Node]:  single lca if it exists, None otherwise 


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/utils.py#L393"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `is_single_common_ancestor`

```python
is_single_common_ancestor(
    graph: Graph,
    candidate: Node,
    nodes: List[Node]
) → bool
```

Determine if a node is the single common ancestor of a list of nodes. 

Note that this function doesn't care about `lowest` property of `lca`. 



**Args:**
  graph (Graph):  graph to perform the check 

 candidate (Node):  node to determine single common ancestor status 

 nodes (List[Node]):  nodes to determine single common ancestor status against 

Returns  bool:  True if `candidate` is a single common ancestor of `nodes`, False otherwise 


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/utils.py#L524"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `find_closest_integer_output_nodes`

```python
find_closest_integer_output_nodes(
    graph: Graph,
    start_nodes: List[Node],
    all_nodes: Dict[Node, NoneType]
) → Tuple[Dict[Node, NoneType], Dict[Node, NoneType]]
```

Find the closest upstream integer output nodes to a set of start nodes in a graph. 



**Args:**
  graph (Graph):  graph to search 

 start_nodes (List[Node]):  nodes from which to start the search 

 all_nodes (Dict[Node, None]):  set of nodes to be extended with visited nodes during the search 



**Returns:**
  Tuple[Dict[Node, None], Dict[Node, None]]:  tuple containing extended `all_nodes` and integer output nodes closest to `start_nodes` 


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/utils.py#L571"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `add_nodes_from_to`

```python
add_nodes_from_to(
    graph: Graph,
    from_nodes: Iterable[Node],
    to_nodes: Dict[Node, NoneType],
    all_nodes: Dict[Node, NoneType]
) → Dict[Node, NoneType]
```

Add nodes from `from_nodes` to `to_nodes`, to `all_nodes`. 



**Args:**
  graph (Graph):  graph to traverse 

 from_nodes (Iterable[Node]):  nodes from which extending `all_nodes` start 

 to_nodes (Dict[Node, None]):  nodes to which extending `all_nodes` stop 

 all_nodes (Dict[Node, None]):  nodes to be extended 



**Returns:**
  Dict[Node, None]:  extended `all_nodes` 


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/utils.py#L619"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `convert_subgraph_to_subgraph_node`

```python
convert_subgraph_to_subgraph_node(
    graph: Graph,
    all_nodes: Dict[Node, NoneType],
    start_nodes: Dict[Node, NoneType],
    terminal_node: Node
) → Optional[Tuple[Node, Node]]
```

Convert a subgraph to Operation.Generic node. 



**Args:**
  graph (Graph):  original graph 

 all_nodes (Dict[Node, None]):  all nodes in the subgraph 

 start_nodes (Dict[Node, None]):  start nodes of the subgraph 

 terminal_node (Node):  terminal node of the subgraph 



**Raises:**
  RuntimeError:  if subgraph is not fusable 



**Returns:**
  Optional[Tuple[Node, Node]]:  None if the subgraph cannot be fused,  subgraph node and its predecessor otherwise 


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/utils.py#L730"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `check_subgraph_fusibility`

```python
check_subgraph_fusibility(
    graph: Graph,
    all_nodes: Dict[Node, NoneType],
    variable_input_node: Node
)
```

Determine if a subgraph can be fused. 

e.g., 

shuffling or reshaping a tensor make fusing impossible as there should be a one-to-one mapping between each cell of the input and each cell of the output for table lookups 



**Args:**
  graph (Graph):  original graph 

 all_nodes (Dict[Node, None]):  all nodes in the subgraph 

 variable_input_node (Node):  variable input node to the subgraph 



**Raises:**
  RuntimeError:  if subgraph is not fusable 


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/utils.py#L791"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `friendly_type_format`

```python
friendly_type_format(type_: type) → str
```

Convert a type to a string. Remove package name and class/type keywords. 


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/utils.py#L809"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_terminal_size`

```python
get_terminal_size() → int
```

Get the terminal size. 


