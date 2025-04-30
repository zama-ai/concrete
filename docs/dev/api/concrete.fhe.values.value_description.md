<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/values/value_description.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.values.value_description`
Declaration of `ValueDescription` class. 



---

<a href="../../frontends/concrete-python/concrete/fhe/values/value_description.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ValueDescription`
ValueDescription class, to combine data type, shape, and encryption status into a single object. 

<a href="../../frontends/concrete-python/concrete/fhe/values/value_description.py#L108"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(dtype: BaseDataType, shape: tuple[int, ], is_encrypted: bool)
```






---

#### <kbd>property</kbd> is_clear

Get whether the value is clear or not. 



**Returns:**
  bool:  True if value is not encrypted, False otherwise 

---

#### <kbd>property</kbd> is_scalar

Get whether the value is scalar or not. 



**Returns:**
  bool:  True if shape of the value is (), False otherwise 

---

#### <kbd>property</kbd> ndim

Get number of dimensions of the value. 



**Returns:**
  int:  number of dimensions of the value 

---

#### <kbd>property</kbd> size

Get number of elements in the value. 



**Returns:**
  int:  number of elements in the value 



---

<a href="../../frontends/concrete-python/concrete/fhe/values/value_description.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `of`

```python
of(value: Any, is_encrypted: bool = False) → ValueDescription
```

Get the `ValueDescription` that can represent `value`. 



**Args:**
  value (Any):  value that needs to be represented 

 is_encrypted (bool, default = False):  whether the resulting `ValueDescription` is encrypted or not 



**Returns:**
  ValueDescription:  `ValueDescription` that can represent `value` 



**Raises:**
  ValueError:  if `value` cannot be represented by `ValueDescription` 


