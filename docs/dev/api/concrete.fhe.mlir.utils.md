<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/mlir/utils.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.mlir.utils`
Declaration of various functions and constants related to MLIR conversion. 

**Global Variables**
---------------
- **MAXIMUM_TLU_BIT_WIDTH**

---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/utils.py#L49"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `flood_replace_none_values`

```python
flood_replace_none_values(table: list)
```

Use flooding algorithm to replace `None` values. 



**Args:**
  table (list):  the list in which there are `None` values that need to be replaced  with copies of the closest non `None` data from the list 


---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/utils.py#L80"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `construct_table_multivariate`

```python
construct_table_multivariate(node: Node, preds: list[Node]) → list[Any]
```

Construct the lookup table for a multivariate node. 



**Args:**
  node (Node):  Multivariate node to construct the table for 

 preds (List[Node]):  ordered predecessors to `node` 



**Returns:**
  List[Any]:  lookup table corresponding to `node` and its input value 


---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/utils.py#L142"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `construct_table`

```python
construct_table(
    node: Node,
    preds: list[Node],
    configuration: Configuration
) → list[Any]
```

Construct the lookup table for an Operation.Generic node. 



**Args:**
  node (Node):  Operation.Generic to construct the table 

 preds (List[Node]):  ordered predecessors to `node` 

 configuration (Configuration):  configuration to use 



**Returns:**
  List[Any]:  lookup table corresponding to `node` and its input value 


---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/utils.py#L276"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `construct_deduplicated_tables`

```python
construct_deduplicated_tables(
    node: Node,
    preds: list[Node],
    configuration: Configuration
) → tuple[tuple[ndarray, Optional[list[tuple[int, ]]]], ]
```

Construct lookup tables for each cell of the input for an Operation.Generic node. 



**Args:**
  node (Node):  Operation.Generic to construct the table 

 preds (List[Node]):  ordered predecessors to `node` 

 configuration (Configuration):  configuration to use 



**Returns:**
  Tuple[Tuple[numpy.ndarray, List[Tuple[int, ...]]], ...]:  tuple containing tuples of 2 for 
            - constructed table 
            - list of indices of the input that use the constructed table 

 e.g., 


 - <b>`.. code-block`</b>: : python 

 (  (np.array([3, 1, 2, 4]), [(1, 0), (2, 1)]),  (np.array([5, 8, 6, 7]), [(0, 0), (0, 1), (1, 1), (2, 0)]),  ) 

means the lookup on 3x2 input will result in 


 - <b>`.. code-block`</b>: : python 

 [ [5, 8, 6, 7][input[0, 0]] , [5, 8, 6, 7][input[0, 1]] ]  [ [3, 1, 2, 4][input[1, 0]] , [5, 8, 6, 7][input[1, 1]] ]  [ [5, 8, 6, 7][input[2, 0]] , [3, 1, 2, 4][input[2, 1]] ] 


---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/utils.py#L32"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `HashableNdarray`
HashableNdarray class, to use numpy arrays in dictionaries. 

<a href="../../frontends/concrete-python/concrete/fhe/mlir/utils.py#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(array: ndarray)
```









---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/utils.py#L374"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Comparison`
Comparison enum, to store the result comparison in 2-bits as there are three possible outcomes. 





