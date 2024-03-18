<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/value_exporter.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.compiler.value_exporter`
ValueExporter. 



---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/value_exporter.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ValueExporter`
A helper class to create `Value`s. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/value_exporter.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(value_exporter: ValueExporter)
```

Wrap the native C++ object. 



**Args:**
  value_exporter (_ValueExporter):  object to wrap 



**Raises:**
  TypeError:  if `value_exporter` is not of type `_ValueExporter` 




---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/value_exporter.py#L52"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `export_scalar`

```python
export_scalar(position: int, value: int) → Value
```

Export scalar. 



**Args:**
  position (int):  position of the argument within the circuit 

 value (int):  scalar to export 



**Returns:**
  Value:  exported scalar 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/value_exporter.py#L70"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `export_tensor`

```python
export_tensor(position: int, values: List[int], shape: List[int]) → Value
```

Export tensor. 



**Args:**
  position (int):  position of the argument within the circuit 

 values (List[int]):  tensor elements to export 

 shape (List[int]):  tensor shape to export 



**Returns:**
  Value:  exported tensor 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/value_exporter.py#L42"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `new`

```python
new(keyset: KeySet, client_parameters: ClientParameters) → ValueExporter
```

Create a value exporter. 


