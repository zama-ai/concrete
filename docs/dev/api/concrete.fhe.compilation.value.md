<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/value.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.compilation.value`
Declaration of `Value` class. 



---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/value.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Value`
Value class, to store scalar or tensor values which can be encrypted or clear. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/value.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(inner: Value)
```








---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/value.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `deserialize`

```python
deserialize(serialized_data: bytes) → Value
```

Deserialize data from bytes. 



**Args:**
  serialized_data (bytes):  previously serialized data 



**Returns:**
  Value:  deserialized data 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/value.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `serialize`

```python
serialize() → bytes
```

Serialize data into bytes. 



**Returns:**
  bytes:  serialized data 


