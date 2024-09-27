<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/client_parameters.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.compiler.client_parameters`
Client parameters. 



---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/client_parameters.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ClientParameters`
ClientParameters are public parameters used for key generation. 

It's a compilation artifact that describes which and how public and private keys should be generated, and used to encrypt arguments of the compiled function. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/client_parameters.py#L26"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(client_parameters: ClientParameters)
```

Wrap the native Cpp object. 



**Args:**
 
 - <b>`client_parameters`</b> (_ClientParameters):  object to wrap 



**Raises:**
 
 - <b>`TypeError`</b>:  if client_parameters is not of type _ClientParameters 




---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/client_parameters.py#L131"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `deserialize`

```python
deserialize(serialized_params: bytes) → ClientParameters
```

Unserialize ClientParameters from bytes of serialized_params. 



**Args:**
 
 - <b>`serialized_params`</b> (bytes):  previously serialized ClientParameters 



**Raises:**
 
 - <b>`TypeError`</b>:  if serialized_params is not of type bytes 



**Returns:**
 
 - <b>`ClientParameters`</b>:  deserialized object 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/client_parameters.py#L115"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `function_list`

```python
function_list() → List[str]
```

Return the list of function names. 



**Returns:**
 
 - <b>`List[str]`</b>:  list of the names of the functions. 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/client_parameters.py#L57"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `input_keyid_at`

```python
input_keyid_at(input_idx: int, circuit_name: str = '<lambda>') → int
```

Get the keyid of a selected encrypted input in a given circuit. 



**Args:**
 
 - <b>`input_idx`</b> (int):  index of the input in the circuit. 
 - <b>`circuit_name`</b> (str):  name of the circuit containing the desired input. 



**Raises:**
 
 - <b>`TypeError`</b>:  if arguments aren't of expected types 



**Returns:**
 
 - <b>`int`</b>:  keyid 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/client_parameters.py#L99"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `input_signs`

```python
input_signs() → List[bool]
```

Return the sign information of inputs. 



**Returns:**
 
 - <b>`List[bool]`</b>:  list of booleans to indicate whether the inputs are signed or not 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/client_parameters.py#L78"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `input_variance_at`

```python
input_variance_at(input_idx: int, circuit_name: str) → float
```

Get the variance of a selected encrypted input in a given circuit. 



**Args:**
 
 - <b>`input_idx`</b> (int):  index of the input in the circuit. 
 - <b>`circuit_name`</b> (str):  name of the circuit containing the desired input. 



**Raises:**
 
 - <b>`TypeError`</b>:  if arguments aren't of expected types 



**Returns:**
 
 - <b>`float`</b>:  variance 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/client_parameters.py#L41"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `lwe_secret_key_param`

```python
lwe_secret_key_param(key_id: int) → LweSecretKeyParam
```

Get the parameters of a selected LWE secret key. 



**Args:**
 
 - <b>`key_id`</b> (int):  keyid to get parameters from 



**Raises:**
 
 - <b>`TypeError`</b>:  if arguments aren't of expected types 



**Returns:**
 
 - <b>`LweSecretKeyParam`</b>:  LWE secret key parameters 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/client_parameters.py#L107"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `output_signs`

```python
output_signs() → List[bool]
```

Return the sign information of outputs. 



**Returns:**
 
 - <b>`List[bool]`</b>:  list of booleans to indicate whether the outputs are signed or not 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/client_parameters.py#L123"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `serialize`

```python
serialize() → bytes
```

Serialize the ClientParameters. 



**Returns:**
 
 - <b>`bytes`</b>:  serialized object 


