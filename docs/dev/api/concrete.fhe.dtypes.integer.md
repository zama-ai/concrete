<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/dtypes/integer.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.dtypes.integer`
Declaration of `Integer` class. 



---

<a href="../../frontends/concrete-python/concrete/fhe/dtypes/integer.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Integer`
Integer class, to represent integers. 

<a href="../../frontends/concrete-python/concrete/fhe/dtypes/integer.py#L110"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(is_signed: bool, bit_width: int)
```








---

<a href="../../frontends/concrete-python/concrete/fhe/dtypes/integer.py#L156"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `can_represent`

```python
can_represent(value: int) → bool
```

Get whether `value` can be represented by the `Integer` or not. 



**Args:**
  value (int):  value to check representability 



**Returns:**
  bool:  True if `value` is representable by the `integer`, False otherwise 

---

<a href="../../frontends/concrete-python/concrete/fhe/dtypes/integer.py#L145"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `max`

```python
max() → int
```

Get the maximum value that can be represented by the `Integer`. 



**Returns:**
  int:  maximum value that can be represented by the `Integer` 

---

<a href="../../frontends/concrete-python/concrete/fhe/dtypes/integer.py#L134"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `min`

```python
min() → int
```

Get the minimum value that can be represented by the `Integer`. 



**Returns:**
  int:  minimum value that can be represented by the `Integer` 

---

<a href="../../frontends/concrete-python/concrete/fhe/dtypes/integer.py#L41"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `that_can_represent`

```python
that_can_represent(value: Any, force_signed: bool = False) → Integer
```

Get the minimal `Integer` that can represent `value`. 



**Args:**
  value (Any):  value that needs to be represented 

 force_signed (bool, default = False):  whether to force signed integers or not 



**Returns:**
  Integer:  minimal `Integer` that can represent `value` 



**Raises:**
  ValueError:  if `value` cannot be represented by `Integer` 

---

<a href="../../frontends/concrete-python/concrete/fhe/dtypes/integer.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update_to_represent`

```python
update_to_represent(value: Any, force_signed: bool = False)
```

Update sign and width inplace to be able to represent `value`. 



**Args:**
  value (Any):  value that needs to be represented 

 force_signed (bool, default = False):  whether to force signed integers or not 



**Raises:**
  ValueError:  if `value` cannot be represented by `Integer` 


