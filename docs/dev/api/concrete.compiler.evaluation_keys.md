<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/evaluation_keys.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.compiler.evaluation_keys`
EvaluationKeys. 



---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/evaluation_keys.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `EvaluationKeys`
EvaluationKeys required for execution. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/evaluation_keys.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(evaluation_keys: EvaluationKeys)
```

Wrap the native Cpp object. 



**Args:**
 
 - <b>`evaluation_keys`</b> (_EvaluationKeys):  object to wrap 



**Raises:**
 
 - <b>`TypeError`</b>:  if evaluation_keys is not of type _EvaluationKeys 




---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/evaluation_keys.py#L43"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `deserialize`

```python
deserialize(serialized_evaluation_keys: bytes) → EvaluationKeys
```

Unserialize EvaluationKeys from bytes. 



**Args:**
 
 - <b>`serialized_evaluation_keys`</b> (bytes):  previously serialized EvaluationKeys 



**Raises:**
 
 - <b>`TypeError`</b>:  if serialized_evaluation_keys is not of type bytes 



**Returns:**
 
 - <b>`EvaluationKeys`</b>:  deserialized object 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/evaluation_keys.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `serialize`

```python
serialize() → bytes
```

Serialize the EvaluationKeys. 



**Returns:**
 
 - <b>`bytes`</b>:  serialized object 


