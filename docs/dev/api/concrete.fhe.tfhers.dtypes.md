<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/dtypes.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.tfhers.dtypes`
Declaration of `TFHERSIntegerType` class. 

**Global Variables**
---------------
- **int8_2_2**
- **uint8_2_2**
- **int16_2_2**
- **uint16_2_2**


---

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/dtypes.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TFHERSIntegerType`
TFHERSIntegerType (Subclass of Integer) to represent tfhers integer types. 

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/dtypes.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(is_signed: bool, bit_width: int, carry_width: int, msg_width: int)
```








---

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/dtypes.py#L67"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `decode`

```python
decode(value: ndarray) → Union[int, ndarray]
```

Decode a tfhers-encoded integer (scalar or tensor). 



**Args:**
 
 - <b>`value`</b> (np.ndarray):  encoded value 



**Raises:**
 
 - <b>`ValueError`</b>:  bad encoding 



**Returns:**
 
 - <b>`Union[int, np.ndarray]`</b>:  decoded value 

---

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/dtypes.py#L40"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `encode`

```python
encode(value: Union[int, integer, ndarray]) → ndarray
```

Encode a scalar or tensor to tfhers integers. 



**Args:**
 
 - <b>`value`</b> (Union[int, np.ndarray]):  scalar or tensor of integer to encode 



**Raises:**
 
 - <b>`TypeError`</b>:  wrong value type 



**Returns:**
 
 - <b>`np.ndarray`</b>:  encoded scalar or tensor 


