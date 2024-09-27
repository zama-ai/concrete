<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/tfhers_int.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.compiler.tfhers_int`
Import and export TFHErs integers into Concrete. 



---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/tfhers_int.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TfhersFheIntDescription`
A helper class to create `TfhersFheIntDescription`s. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/tfhers_int.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(desc: TfhersFheIntDescription)
```

Wrap the native C++ object. 



**Args:**
  desc (_TfhersFheIntDescription):  object to wrap 



**Raises:**
  TypeError:  if `desc` is not of type `_TfhersFheIntDescription` 


---

#### <kbd>property</kbd> carry_modulus

Modulus of the carry part in each ciphertext 

---

#### <kbd>property</kbd> degree

Tracks the number of operations that have been done 

---

#### <kbd>property</kbd> is_signed

Is the integer signed 

---

#### <kbd>property</kbd> ks_first

Keyswitch placement relative to the bootsrap in a PBS 

---

#### <kbd>property</kbd> lwe_size

LWE size 

---

#### <kbd>property</kbd> message_modulus

Modulus of the message part in each ciphertext 

---

#### <kbd>property</kbd> n_cts

Number of ciphertexts 

---

#### <kbd>property</kbd> noise_level

Noise level 

---

#### <kbd>property</kbd> width

Total integer bitwidth 



---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/tfhers_int.py#L181"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `from_serialized_fheuint8`

```python
from_serialized_fheuint8(buffer: bytes) → TfhersFheIntDescription
```

Get the description of a serialized TFHErs fheuint8. 



**Args:**
 
 - <b>`buffer`</b> (bytes):  serialized fheuint8 



**Raises:**
 
 - <b>`TypeError`</b>:  buffer is not of type bytes 



**Returns:**
 
 - <b>`TfhersFheIntDescription`</b>:  description of the serialized fheuint8 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/tfhers_int.py#L155"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_unknown_noise_level`

```python
get_unknown_noise_level() → int
```

Get unknow noise level value. 



**Returns:**
 
 - <b>`int`</b>:  unknown noise level value 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/tfhers_int.py#L40"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `new`

```python
new(
    width: int,
    is_signed: bool,
    message_modulus: int,
    carry_modulus: int,
    degree: int,
    lwe_size: int,
    n_cts: int,
    noise_level: int,
    ks_first: bool
) → TfhersFheIntDescription
```

Create a TfhersFheIntDescription. 



**Args:**
 
 - <b>`width`</b> (int):  integer width 
 - <b>`is_signed`</b> (bool):  signed or unsigned 
 - <b>`message_modulus`</b> (int):  message modulus (not its log2) 
 - <b>`carry_modulus`</b> (int):  carry modulus (not its log2) 
 - <b>`degree`</b> (int):  degree 
 - <b>`lwe_size`</b> (int):  LWE size 
 - <b>`n_cts`</b> (int):  number of ciphertexts 
 - <b>`noise_level`</b> (int):  noise level 
 - <b>`ks_first`</b> (bool):  PBS order (keyswitch first, or bootstrap first) 



**Returns:**
 
 - <b>`TfhersFheIntDescription`</b>:  TFHErs integer description 


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/tfhers_int.py#L199"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TfhersExporter`
A helper class to import and export TFHErs big integers. 




---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/tfhers_int.py#L202"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `export_fheuint8`

```python
export_fheuint8(value: Value, info: TfhersFheIntDescription) → bytes
```

Convert Concrete value to TFHErs and serialize it. 



**Args:**
 
 - <b>`value`</b> (Value):  value to export 
 - <b>`info`</b> (TfhersFheIntDescription):  description of the TFHErs integer to export to 



**Raises:**
 
 - <b>`TypeError`</b>:  if wrong input types 



**Returns:**
 
 - <b>`bytes`</b>:  converted and serialized fheuint8 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/tfhers_int.py#L224"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `import_fheuint8`

```python
import_fheuint8(
    buffer: bytes,
    info: TfhersFheIntDescription,
    keyid: int,
    variance: float
) → Value
```

Unserialize and convert from TFHErs to Concrete value. 



**Args:**
 
 - <b>`buffer`</b> (bytes):  serialized fheuint8 
 - <b>`info`</b> (TfhersFheIntDescription):  description of the TFHErs integer to import 
 - <b>`keyid`</b> (int):  id of the key used for encryption 
 - <b>`variance`</b> (float):  variance used for encryption 



**Raises:**
 
 - <b>`TypeError`</b>:  if wrong input types 



**Returns:**
 
 - <b>`Value`</b>:  unserialized and converted value 


