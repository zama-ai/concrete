<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/client_parameters.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.compiler.client_parameters`
Client parameters. 



---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/client_parameters.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ClientParameters`
ClientParameters are public parameters used for key generation. 

It's a compilation artifact that describes which and how public and private keys should be generated, and used to encrypt arguments of the compiled function. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/client_parameters.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/client_parameters.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/client_parameters.py#L40"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `input_signs`

```python
input_signs() → List[bool]
```

Return the sign information of inputs. 



**Returns:**
 
 - <b>`List[bool]`</b>:  list of booleans to indicate whether the inputs are signed or not 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/client_parameters.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `output_signs`

```python
output_signs() → List[bool]
```

Return the sign information of outputs. 



**Returns:**
 
 - <b>`List[bool]`</b>:  list of booleans to indicate whether the outputs are signed or not 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/client_parameters.py#L56"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `serialize`

```python
serialize() → bytes
```

Serialize the ClientParameters. 



**Returns:**
 
 - <b>`bytes`</b>:  serialized object 


