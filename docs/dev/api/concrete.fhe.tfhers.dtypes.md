<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/dtypes.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.tfhers.dtypes`
Declaration of `TFHERSIntegerType` class. 



---

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/dtypes.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `EncryptionKeyChoice`
TFHErs key choice: big or small. 





---

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/dtypes.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CryptoParams`
Crypto parameters used for a tfhers integer. 

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/dtypes.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    lwe_dimension: int,
    glwe_dimension: int,
    polynomial_size: int,
    pbs_base_log: int,
    pbs_level: int,
    lwe_noise_distribution: float,
    glwe_noise_distribution: float,
    encryption_key_choice: EncryptionKeyChoice
)
```








---

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/dtypes.py#L94"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `encryption_variance`

```python
encryption_variance() → float
```

Get encryption variance based on parameters. 

This will return different values depending on the encryption key choice. 



**Returns:**
 
 - <b>`float`</b>:  encryption variance 

---

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/dtypes.py#L71"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `from_dict`

```python
from_dict(dict_obj: dict[str, Any]) → CryptoParams
```

Create a CryptoParams instance from a dictionary. 



**Args:**
 
 - <b>`dict_obj`</b> (dict):  A dictionary containing the parameters. 



**Returns:**
 CryptoParams:  An instance of CryptoParams initialized with the values from the dictionary. 

---

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/dtypes.py#L53"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Convert the CryptoParams object to a dictionary representation. 



**Returns:**
 
 - <b>`Dict[str, Any]`</b>:  dictionary representation 


---

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/dtypes.py#L141"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TFHERSIntegerType`
TFHERSIntegerType (Subclass of Integer) to represent tfhers integer types. 

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/dtypes.py#L150"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    is_signed: bool,
    bit_width: int,
    carry_width: int,
    msg_width: int,
    params: CryptoParams
)
```








---

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/dtypes.py#L250"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `decode`

```python
decode(value: Union[list, ndarray]) → Union[int, ndarray]
```

Decode a tfhers-encoded integer (scalar or tensor). 



**Args:**
 
 - <b>`value`</b> (np.ndarray):  encoded value 



**Raises:**
 
 - <b>`ValueError`</b>:  bad encoding 



**Returns:**
 
 - <b>`Union[int, np.ndarray]`</b>:  decoded value 

---

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/dtypes.py#L212"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `encode`

```python
encode(value: Union[int, integer, list, ndarray]) → ndarray
```

Encode a scalar or tensor to tfhers integers. 



**Args:**
 
 - <b>`value`</b> (Union[int, np.ndarray]):  scalar or tensor of integer to encode 



**Raises:**
 
 - <b>`TypeError`</b>:  wrong value type 



**Returns:**
 
 - <b>`np.ndarray`</b>:  encoded scalar or tensor 

---

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/dtypes.py#L178"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `from_dict`

```python
from_dict(dict_obj) → TFHERSIntegerType
```

Create a TFHERSIntegerType instance from a dictionary. 



**Args:**
 
 - <b>`dict_obj`</b> (dict):  A dictionary representation of the object. 



**Returns:**
 
 - <b>`TFHERSIntegerType`</b>:  An instance of TFHERSIntegerType created from the dictionary. 

---

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/dtypes.py#L163"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `to_dict`

```python
to_dict() → dict[str, Any]
```

Convert the object to a dictionary representation. 



**Returns:**
 
 - <b>`Dict[str, Any]`</b>:  A dictionary containing the object's attributes 


