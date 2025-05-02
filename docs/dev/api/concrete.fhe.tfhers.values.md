<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/values.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.tfhers.values`
Declaration of `TFHERSInteger` which wraps values as being of tfhers types. 



---

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/values.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TFHERSInteger`
Wrap integer values (scalar or arrays) into typed values, using tfhers types. 

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/values.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(dtype: TFHERSIntegerType, value: Union[list, int, ndarray])
```






---

#### <kbd>property</kbd> dtype

Get the type of the wrapped value. 



**Returns:**
  TFHERSIntegerType 

---

#### <kbd>property</kbd> shape

Get the shape of the wrapped value. 



**Returns:**
 
 - <b>`tuple`</b>:  shape 

---

#### <kbd>property</kbd> value

Get the wrapped value. 



**Returns:**
  Union[int, np.ndarray] 



---

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/values.py#L88"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `max`

```python
max()
```

Get the maximum value that can be represented by the current type. 



**Returns:**
  int:  maximum value that can be represented by the current type 

---

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/values.py#L78"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `min`

```python
min()
```

Get the minimum value that can be represented by the current type. 



**Returns:**
  int:  minimum value that can be represented by the current type 


