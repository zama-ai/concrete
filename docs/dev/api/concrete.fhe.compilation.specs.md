<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/compilation/specs.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.compilation.specs`
Declaration of `ClientSpecs` class. 



---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/specs.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ClientSpecs`
ClientSpecs class, to create Client objects. 

<a href="../../frontends/concrete-python/concrete/fhe/compilation/specs.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(client_parameters: ClientParameters)
```








---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/specs.py#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `deserialize`

```python
deserialize(serialized_client_specs: bytes) → ClientSpecs
```

Create client specs from its string representation. 



**Args:**
  serialized_client_specs (bytes):  client specs to deserialize 



**Returns:**
  ClientSpecs:  deserialized client specs 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/specs.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `serialize`

```python
serialize() → bytes
```

Serialize client specs into a string representation. 



**Returns:**
  bytes:  serialized client specs 


