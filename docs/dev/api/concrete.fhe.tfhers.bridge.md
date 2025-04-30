<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/bridge.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.tfhers.bridge`
Declaration of `tfhers.Bridge` class. 


---

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/bridge.py#L303"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `new_bridge`

```python
new_bridge(
    circuit_or_module_or_client: Union[ForwardRef('Circuit'), ForwardRef('Module'), ForwardRef('Client')]
) → Bridge
```

Create a TFHErs bridge from a circuit or module or client. 



**Args:**
  client (Union["fhe.Circuit", "fhe.Module", "fhe.Client"]):  The client|circuit|module instance to be used by the Bridge. 



**Returns:**
 
 - <b>`Bridge`</b>:  A new Bridge instance attached to the provided client. 


---

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/bridge.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Bridge`
TFHErs Bridge extend a Client with TFHErs functionalities. 

client (fhe.Client): the client instance to be attached by the Bridge tfhers_specs (fhe.tfhers.TFHERSClientSpecs): the TFHE-rs specs of the client 

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/bridge.py#L29"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(client: 'Client')
```






---

#### <kbd>property</kbd> input_shapes_per_func

Return the input shapes per function map. 

---

#### <kbd>property</kbd> input_types_per_func

Return the input types per function map. 

---

#### <kbd>property</kbd> output_shapes_per_func

Return the output shapes per function map. 

---

#### <kbd>property</kbd> output_types_per_func

Return the output types per function map. 



---

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/bridge.py#L190"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `export_value`

```python
export_value(
    value: Value,
    output_idx: int,
    func_name: Optional[str] = None
) → bytes
```

Export a value as a serialized TFHErs integer. 



**Args:**
 
 - <b>`value`</b> (TransportValue):  value to export 
 - <b>`output_idx`</b> (int):  the index corresponding to this output 
 - <b>`func_name`</b> (Optional[str]):  name of the function the value belongs to.  Doesn't need to be provided if there is a single function. 



**Returns:**
 
 - <b>`bytes`</b>:  serialized fheuint8 

---

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/bridge.py#L164"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `import_value`

```python
import_value(
    buffer: bytes,
    input_idx: int,
    func_name: Optional[str] = None
) → Value
```

Import a serialized TFHErs integer as a Value. 



**Args:**
 
 - <b>`buffer`</b> (bytes):  serialized integer 
 - <b>`input_idx`</b> (int):  the index of the input expecting this value 
 - <b>`func_name`</b> (Optional[str]):  name of the function the value belongs to.  Doesn't need to be provided if there is a single function. 



**Returns:**
 
 - <b>`fhe.TransportValue`</b>:  imported value 

---

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/bridge.py#L237"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `keygen_with_initial_keys`

```python
keygen_with_initial_keys(
    input_idx_to_key_buffer: dict[Union[tuple[str, int], int], bytes],
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

 input_idx_to_key_buffer (Dict[Union[Tuple[str, int], int], bytes]): 
 - <b>`initial keys to set before keygen. Two possible formats`</b>:  the first is when you have a single function. Here you can just provide the position of the input as index. The second is when you have multiple functions. You will need to provide both the name of the function and the input's position as index. 



**Raises:**
 
 - <b>`RuntimeError`</b>:  if failed to deserialize the key 

---

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/bridge.py#L215"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `serialize_input_secret_key`

```python
serialize_input_secret_key(
    input_idx: int,
    func_name: Optional[str] = None
) → bytes
```

Serialize secret key used for a specific input. 



**Args:**
 
 - <b>`input_idx`</b> (int):  input index corresponding to the key to serialize 
 - <b>`func_name`</b> (Optional[str]):  name of the function the key belongs to.  Doesn't need to be provided if there is a single function. 



**Returns:**
 
 - <b>`bytes`</b>:  serialized key 


