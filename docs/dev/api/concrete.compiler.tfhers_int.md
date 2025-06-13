<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/tfhers_int.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.compiler.tfhers_int`
Import and export TFHErs integers into Concrete. 



---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/tfhers_int.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TfhersExporter`
A helper class to import and export TFHErs big integers. 




---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/tfhers_int.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `export_int`

```python
export_int(value: TransportValue, info: TfhersFheIntDescription) → bytes
```

Convert Concrete value to TFHErs and serialize it. 



**Args:**
 


 - <b>`value`</b> (Value):  value to export 
 - <b>`info`</b> (TfhersFheIntDescription):  description of the TFHErs integer to export to 



**Raises:**
 
 - <b>`TypeError`</b>:  if wrong input types 



**Returns:**
 
 - <b>`bytes`</b>:  converted and serialized TFHErs integer 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/tfhers_int.py#L43"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `import_int`

```python
import_int(
    buffer: bytes,
    info: TfhersFheIntDescription,
    keyid: int,
    variance: float,
    shape: Tuple[int, ]
) → TransportValue
```

Unserialize and convert from TFHErs to Concrete value. 



**Args:**
 
 - <b>`buffer`</b> (bytes):  serialized TFHErs integer 
 - <b>`info`</b> (TfhersFheIntDescription):  description of the TFHErs integer to import 
 - <b>`keyid`</b> (int):  id of the key used for encryption 
 - <b>`variance`</b> (float):  variance used for encryption 
 - <b>`shape`</b> (Tuple[int, ...]):  expected shape 



**Raises:**
 
 - <b>`TypeError`</b>:  if wrong input types 



**Returns:**
 
 - <b>`TransportValue`</b>:  unserialized and converted value 


