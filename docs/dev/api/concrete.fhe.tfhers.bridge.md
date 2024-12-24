<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/bridge.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.tfhers.bridge`
Declaration of `tfhers.Bridge` class. 


---

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/bridge.py#L292"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `new_bridge`

```python
new_bridge(
    circuit_or_module: Union[ForwardRef('Circuit'), ForwardRef('Module')]
) → Bridge
```

Create a TFHErs bridge from a circuit or module. 



**Args:**
 
 - <b>`circuit`</b> (Union[Circuit, Module]):  compiled circuit or module 



**Returns:**
 
 - <b>`Bridge`</b>:  TFHErs bridge 


---

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/bridge.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Bridge`
TFHErs Bridge extend an Module with TFHErs functionalities. 

input_types_per_func (Dict[str, List[Optional[TFHERSIntegerType]]]):  maps every input to a type for every function in the module. None means a non-tfhers type output_types_per_func (Dict[str, List[Optional[TFHERSIntegerType]]]):  maps every output to a type for every function in the module. None means a non-tfhers type input_shapes_per_func (Dict[str, List[Optional[Tuple[int, ...]]]]):  maps every input to a shape for every function in the module. None means a non-tfhers type output_shapes_per_func (Dict[str, List[Optional[Tuple[int, ...]]]]):  maps every output to a shape for every function in the module. None means a non-tfhers type 

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/bridge.py#L36"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    module: 'Module',
    input_types_per_func: Dict[str, List[Optional[TFHERSIntegerType]]],
    output_types_per_func: Dict[str, List[Optional[TFHERSIntegerType]]],
    input_shapes_per_func: Dict[str, List[Optional[Tuple[int, ]]]],
    output_shapes_per_func: Dict[str, List[Optional[Tuple[int, ]]]]
)
```








---

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/bridge.py#L180"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/bridge.py#L154"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/bridge.py#L227"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `keygen_with_initial_keys`

```python
keygen_with_initial_keys(
    input_idx_to_key_buffer: Dict[Union[Tuple[str, int], int], bytes],
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

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/bridge.py#L205"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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


