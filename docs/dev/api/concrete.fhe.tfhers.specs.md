<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/specs.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.tfhers.specs`
TFHE-rs client specs. 



---

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/specs.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TFHERSClientSpecs`
TFHE-rs client specs. 

Contains info about TFHE-rs inputs and outputs. 

input_types_per_func (Dict[str, List[Optional[TFHERSIntegerType]]]):  maps every input to a type for every function in the module. None means a non-tfhers type output_types_per_func (Dict[str, List[Optional[TFHERSIntegerType]]]):  maps every output to a type for every function in the module. None means a non-tfhers type input_shapes_per_func (Dict[str, List[Optional[Tuple[int, ...]]]]):  maps every input to a shape for every function in the module. None means a non-tfhers type output_shapes_per_func (Dict[str, List[Optional[Tuple[int, ...]]]]):  maps every output to a shape for every function in the module. None means a non-tfhers type 

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/specs.py#L29"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    input_types_per_func: dict[str, list[Optional[TFHERSIntegerType]]],
    output_types_per_func: dict[str, list[Optional[TFHERSIntegerType]]],
    input_shapes_per_func: dict[str, list[Optional[tuple[int, ]]]],
    output_shapes_per_func: dict[str, list[Optional[tuple[int, ]]]]
)
```








---

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/specs.py#L117"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `from_dict`

```python
from_dict(dict_obj: dict[str, Any]) → TFHERSClientSpecs
```

Create a TFHERSClientSpecs instance from a dictionary. 



**Args:**
 
 - <b>`dict_obj`</b> (Dict[str, Any]):  A dictionary containing the specifications. 



**Returns:**
 
 - <b>`TFHERSClientSpecs`</b>:  An instance of TFHERSClientSpecs created from the dictionary. 

---

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/specs.py#L49"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `from_graphs`

```python
from_graphs(graphs: dict[str, Graph]) → TFHERSClientSpecs
```

Create a TFHERSClientSpecs instance from a dictionary of graphs. 



**Args:**
 
 - <b>`graphs`</b> (Dict[str, Graph]):  graphs to extract the specs from 

**Returns:**
 
 - <b>`TFHERSClientSpecs`</b>:  An instance of TFHERSClientSpecs containing the input  and output types and shapes for each function. 

---

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/specs.py#L97"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Convert the TFHERSClientSpecs object to a dictionary representation. 



**Returns:**
 
 - <b>`Dict[str, Any]`</b>:  dictionary representation 


