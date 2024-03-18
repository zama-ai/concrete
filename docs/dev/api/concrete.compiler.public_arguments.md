<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/public_arguments.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.compiler.public_arguments`
PublicArguments. 



---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/public_arguments.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PublicArguments`
PublicArguments holds encrypted and plain arguments, as well as public materials. 

An encrypted computation may require both encrypted and plain arguments, PublicArguments holds both types, but also other public materials, such as public keys, which are required for private computation. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/public_arguments.py#L26"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(public_arguments: PublicArguments)
```

Wrap the native Cpp object. 



**Args:**
 
 - <b>`public_arguments`</b> (_PublicArguments):  object to wrap 



**Raises:**
 
 - <b>`TypeError`</b>:  if public_arguments is not of type _PublicArguments 




---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/public_arguments.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `deserialize`

```python
deserialize(
    client_parameters: ClientParameters,
    serialized_args: bytes
) → PublicArguments
```

Unserialize PublicArguments from bytes of serialized_args. 



**Args:**
 
 - <b>`client_parameters`</b> (ClientParameters):  client parameters of the compiled circuit 
 - <b>`serialized_args`</b> (bytes):  previously serialized PublicArguments 



**Raises:**
 
 - <b>`TypeError`</b>:  if client_parameters is not of type ClientParameters 
 - <b>`TypeError`</b>:  if serialized_args is not of type bytes 



**Returns:**
 
 - <b>`PublicArguments`</b>:  deserialized object 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/public_arguments.py#L41"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `new`

```python
new(client_parameters: ClientParameters, values: List[Value]) → PublicArguments
```

Create public arguments from individual values. 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/public_arguments.py#L56"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `serialize`

```python
serialize() → bytes
```

Serialize the PublicArguments. 



**Returns:**
 
 - <b>`bytes`</b>:  serialized object 


