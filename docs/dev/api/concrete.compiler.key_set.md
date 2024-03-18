<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/key_set.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.compiler.key_set`
KeySet. 

Store for the different keys required for an encrypted computation. 



---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/key_set.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `KeySet`
KeySet stores the different keys required for an encrypted computation. 

Holds private keys (secret key) used for encryption/decryption, and public keys used for computation. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/key_set.py#L26"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(keyset: KeySet)
```

Wrap the native Cpp object. 



**Args:**
 
 - <b>`keyset`</b> (_KeySet):  object to wrap 



**Raises:**
 
 - <b>`TypeError`</b>:  if keyset is not of type _KeySet 




---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/key_set.py#L47"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `deserialize`

```python
deserialize(serialized_key_set: bytes) → KeySet
```

Deserialize KeySet from bytes. 



**Args:**
 
 - <b>`serialized_key_set`</b> (bytes):  previously serialized KeySet 



**Raises:**
 
 - <b>`TypeError`</b>:  if serialized_key_set is not of type bytes 



**Returns:**
 
 - <b>`KeySet`</b>:  deserialized object 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/key_set.py#L66"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_evaluation_keys`

```python
get_evaluation_keys() → EvaluationKeys
```

Get evaluation keys for execution. 



**Returns:**
  EvaluationKeys:  evaluation keys for execution 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/key_set.py#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `serialize`

```python
serialize() → bytes
```

Serialize the KeySet. 



**Returns:**
 
 - <b>`bytes`</b>:  serialized object 


