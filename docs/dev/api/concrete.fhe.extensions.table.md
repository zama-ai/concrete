<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/extensions/table.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.extensions.table`
Declaration of `LookupTable` class. 



---

<a href="../../frontends/concrete-python/concrete/fhe/extensions/table.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LookupTable`
LookupTable class, to provide a way to do direct table lookups. 

<a href="../../frontends/concrete-python/concrete/fhe/extensions/table.py#L23"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(table: Any)
```








---

<a href="../../frontends/concrete-python/concrete/fhe/extensions/table.py#L93"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `apply`

```python
apply(
    key: Union[int, integer, bool_, ndarray],
    table: ndarray
) → Union[int, integer, ndarray]
```

Apply lookup table. 



**Args:**
  key (Union[int, np.integer, np.bool_, np.ndarray]):  lookup key 

 table (np.ndarray):  lookup table 



**Returns:**
  Union[int, np.integer, np.ndarray]:  lookup result 



**Raises:**
  ValueError:  if `table` cannot be looked up with `key` 


