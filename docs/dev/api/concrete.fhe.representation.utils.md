<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/representation/utils.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.representation.utils`
Declaration of various functions and constants related to representation of computation. 

**Global Variables**
---------------
- **KWARGS_IGNORED_IN_FORMATTING**
- **SPECIAL_OBJECT_MAPPING**
- **NODES_THAT_HAVE_TLU_WHEN_ALL_INPUTS_ARE_ENCRYPTED**

---

<a href="../../frontends/concrete-python/concrete/fhe/representation/utils.py#L65"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `format_constant`

```python
format_constant(
    constant: Any,
    maximum_length: int = 45,
    keep_newlines: bool = False
) → str
```

Get the textual representation of a constant. 



**Args:**
  constant (Any):  constant to format 

 maximum_length (int, default = 45):  maximum length of the resulting string 

 keep_newlines (bool, default = False):  whether to keep newlines or not 



**Returns:**
  str:  textual representation of `constant` 


---

<a href="../../frontends/concrete-python/concrete/fhe/representation/utils.py#L108"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `format_indexing_element`

```python
format_indexing_element(indexing_element: Union[int, integer, slice, Any])
```

Format an indexing element. 

This is required mainly for slices. The reason is that string representation of slices are very long and verbose. To give an example, `x[:, 2:]` will have the following index `[slice(None, None, None), slice(2, None, None)]` if printed naively. With this helper, it will be formatted as `[:, 2:]`. 



**Args:**
  indexing_element (Union[int, np.integer, slice]):  indexing element to format 



**Returns:**
  str:  textual representation of `indexing_element` 


