<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/value_decrypter.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.compiler.value_decrypter`
ValueDecrypter. 



---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/value_decrypter.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ValueDecrypter`
A helper class to decrypt `Value`s. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/value_decrypter.py#L23"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(value_decrypter: ValueDecrypter)
```

Wrap the native C++ object. 



**Args:**
  value_decrypter (_ValueDecrypter):  object to wrap 



**Raises:**
  TypeError:  if `value_decrypter` is not of type `_ValueDecrypter` 




---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/value_decrypter.py#L53"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `decrypt`

```python
decrypt(position: int, value: Value) → Union[int, ndarray]
```

Decrypt value. 



**Args:**
  position (int):  position of the argument within the circuit 

 value (Value):  value to decrypt 



**Returns:**
  Union[int, np.ndarray]:  decrypted value 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/value_decrypter.py#L43"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `new`

```python
new(keyset: KeySet, client_parameters: ClientParameters)
```

Create a value decrypter. 


