<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/dtypes.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.tfhers.dtypes`
Declaration of `TFHERSIntegerType` class. 

**Global Variables**
---------------
- **int8**
- **uint8**
- **int16**
- **uint16**
- **int8_2_2**
- **uint8_2_2**
- **int16_2_2**
- **uint16_2_2**


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

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/dtypes.py#L53"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `encryption_variance`

```python
encryption_variance() → float
```

Get encryption variance based on parameters. 

This will return different values depending on the encryption key choice. 



**Returns:**
 
 - <b>`float`</b>:  encryption variance 


---

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/dtypes.py#L97"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TFHERSIntegerType`
TFHERSIntegerType (Subclass of Integer) to represent tfhers integer types. 

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/dtypes.py#L106"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/dtypes.py#L172"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/dtypes.py#L134"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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


