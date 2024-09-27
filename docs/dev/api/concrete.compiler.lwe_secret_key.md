<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/lwe_secret_key.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.compiler.lwe_secret_key`
LweSecretKey. 



---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/lwe_secret_key.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LweSecretKeyParam`
LWE Secret Key Parameters 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/lwe_secret_key.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(lwe_secret_key_param: LweSecretKeyParam)
```

Wrap the native Cpp object. 



**Args:**
 
 - <b>`lwe_secret_key_param`</b> (_LweSecretKeyParam):  object to wrap 



**Raises:**
 
 - <b>`TypeError`</b>:  if lwe_secret_key_param is not of type _LweSecretKeyParam 


---

#### <kbd>property</kbd> dimension

LWE dimension 




---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/lwe_secret_key.py#L42"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LweSecretKey`
An LweSecretKey. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/lwe_secret_key.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(lwe_secret_key: LweSecretKey)
```

Wrap the native Cpp object. 



**Args:**
 
 - <b>`lwe_secret_key`</b> (_LweSecretKey):  object to wrap 



**Raises:**
 
 - <b>`TypeError`</b>:  if lwe_secret_key is not of type _LweSecretKey 


---

#### <kbd>property</kbd> param

LWE Secret Key Parameters 



---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/lwe_secret_key.py#L69"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `deserialize`

```python
deserialize(serialized_key: bytes, param: LweSecretKeyParam) → LweSecretKey
```

Deserialize LweSecretKey from bytes. 



**Args:**
 
 - <b>`serialized_key`</b> (bytes):  previously serialized secret key 



**Raises:**
 
 - <b>`TypeError`</b>:  if wrong types for input arguments 



**Returns:**
 
 - <b>`LweSecretKey`</b>:  deserialized object 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/lwe_secret_key.py#L111"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `deserialize_from_glwe`

```python
deserialize_from_glwe(
    serialized_glwe_key: bytes,
    param: LweSecretKeyParam
) → LweSecretKey
```

Deserialize LweSecretKey from glwe secret key bytes. 



**Args:**
 
 - <b>`serialized_glwe_key`</b> (bytes):  previously serialized glwe secret key 



**Raises:**
 
 - <b>`TypeError`</b>:  if wrong types for input arguments 



**Returns:**
 
 - <b>`LweSecretKey`</b>:  deserialized object 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/lwe_secret_key.py#L60"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `serialize`

```python
serialize() → bytes
```

Serialize key. 



**Returns:**
 
 - <b>`bytes`</b>:  serialized key 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/lwe_secret_key.py#L92"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `serialize_as_glwe`

```python
serialize_as_glwe(glwe_dim: int, poly_size: int) → bytes
```

Serialize key as a glwe secret key. 



**Args:**
 
 - <b>`glwe_dim`</b> (int):  glwe dimension of the key 
 - <b>`poly_size`</b> (int):  polynomial size of the key 



**Raises:**
 
 - <b>`TypeError`</b>:  if wrong types for input arguments 



**Returns:**
 
 - <b>`bytes`</b>:  serialized key 


