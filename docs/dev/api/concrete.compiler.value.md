<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/value.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.compiler.value`
Value. 



---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/value.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Value`
An encrypted/clear value which can be scalar/tensor. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/value.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(value: Value)
```

Wrap the native C++ object. 



**Args:**
  value (_Value):  object to wrap 



**Raises:**
  TypeError:  if `value` is not of type `_Value` 




---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/value.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `deserialize`

```python
deserialize(serialized_value: bytes) → Value
```

Deserialize value from bytes. 



**Args:**
  serialized_value (bytes):  previously serialized value 



**Returns:**
  Value:  deserialized value 



**Raises:**
  TypeError:  if `serialized_value` is not of type `bytes` 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/value.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `serialize`

```python
serialize() → bytes
```

Serialize value into bytes. 



**Returns:**
 
 - <b>`bytes`</b>:  serialized value 


