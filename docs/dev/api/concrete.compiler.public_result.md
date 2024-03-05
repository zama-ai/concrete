<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/public_result.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.compiler.public_result`
PublicResult. 



---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/public_result.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PublicResult`
PublicResult holds the result of an encrypted execution and can be decrypted using ClientSupport. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/public_result.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(public_result: PublicResult)
```

Wrap the native Cpp object. 



**Args:**
 
 - <b>`public_result`</b> (_PublicResult):  object to wrap 



**Raises:**
 
 - <b>`TypeError`</b>:  if public_result is not of type _PublicResult 




---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/public_result.py#L55"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `deserialize`

```python
deserialize(
    client_parameters: ClientParameters,
    serialized_result: bytes
) → PublicResult
```

Unserialize PublicResult from bytes of serialized_result. 



**Args:**
 
 - <b>`client_parameters`</b> (ClientParameters):  client parameters of the compiled circuit 
 - <b>`serialized_result`</b> (bytes):  previously serialized PublicResult 



**Raises:**
 
 - <b>`TypeError`</b>:  if client_parameters is not of type ClientParameters 
 - <b>`TypeError`</b>:  if serialized_result is not of type bytes 



**Returns:**
 
 - <b>`PublicResult`</b>:  deserialized object 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/public_result.py#L41"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_value`

```python
get_value(position: int) → Value
```

Get a specific value in the result. 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/public_result.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `n_values`

```python
n_values() → int
```

Get number of values in the result. 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/public_result.py#L47"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `serialize`

```python
serialize() → bytes
```

Serialize the PublicResult. 



**Returns:**
 
 - <b>`bytes`</b>:  serialized object 


