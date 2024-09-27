<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/bridge.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.tfhers.bridge`
Declaration of `tfhers.Bridge` class. 


---

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/bridge.py#L232"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `new_bridge`

```python
new_bridge(circuit: 'Circuit') → Bridge
```

Create a TFHErs bridge from a circuit. 



**Args:**
 
 - <b>`circuit`</b> (Circuit):  compiled circuit 



**Returns:**
 
 - <b>`Bridge`</b>:  TFHErs bridge 


---

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/bridge.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Bridge`
TFHErs Bridge extend a Circuit with TFHErs functionalities. 

input_types (List[Optional[TFHERSIntegerType]]): maps every input to a type. None means  a non-tfhers type output_types (List[Optional[TFHERSIntegerType]]): maps every output to a type. None means  a non-tfhers type 

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/bridge.py#L27"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    circuit: 'Circuit',
    input_types: List[Optional[TFHERSIntegerType]],
    output_types: List[Optional[TFHERSIntegerType]]
)
```








---

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/bridge.py#L135"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `export_value`

```python
export_value(value: 'Value', output_idx: int) → bytes
```

Export a value as a serialized TFHErs integer. 



**Args:**
 
 - <b>`value`</b> (fhe.Value):  value to export 
 - <b>`output_idx`</b> (int):  the index corresponding to this output 



**Returns:**
 
 - <b>`bytes`</b>:  serialized fheuint8 

---

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/bridge.py#L102"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `import_value`

```python
import_value(buffer: bytes, input_idx: int) → Value
```

Import a serialized TFHErs integer as a Value. 



**Args:**
 
 - <b>`buffer`</b> (bytes):  serialized integer 
 - <b>`input_idx`</b> (int):  the index of the input expecting this value 



**Returns:**
 
 - <b>`fhe.Value`</b>:  imported value 

---

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/bridge.py#L179"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `keygen_with_initial_keys`

```python
keygen_with_initial_keys(
    input_idx_to_key_buffer: Dict[int, bytes],
    force: bool = False,
    seed: Optional[int] = None,
    encryption_seed: Optional[int] = None
)
```

Generate keys using an initial set of secret keys. 



**Args:**
  force (bool, default = False):  whether to generate new keys even if keys are already generated 

 seed (Optional[int], default = None):  seed for private keys randomness 

 encryption_seed (Optional[int], default = None):  seed for encryption randomness 


 - <b>`input_idx_to_key_buffer`</b> (Dict[int, bytes]):  initial keys to set before keygen 



**Raises:**
 
 - <b>`RuntimeError`</b>:  if failed to deserialize the key 

---

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/bridge.py#L164"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `serialize_input_secret_key`

```python
serialize_input_secret_key(input_idx: int) → bytes
```

Serialize secret key used for a specific input. 



**Args:**
 
 - <b>`input_idx`</b> (int):  input index corresponding to the key to serialize 



**Returns:**
 
 - <b>`bytes`</b>:  serialized key 


