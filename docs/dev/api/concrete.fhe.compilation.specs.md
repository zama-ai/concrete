<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/compilation/specs.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.compilation.specs`
Declaration of `ClientSpecs` class. 

**Global Variables**
---------------
- **TFHERS_SPECS_KEY**


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/specs.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ClientSpecs`
ClientSpecs class, to create Client objects. 

<a href="../../frontends/concrete-python/concrete/fhe/compilation/specs.py#L26"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    program_info: ProgramInfo,
    tfhers_specs: Optional[ForwardRef('TFHERSClientSpecs')] = None
)
```








---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/specs.py#L53"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `deserialize`

```python
deserialize(serialized_client_specs: bytes) → ClientSpecs
```

Create client specs from bytes. 



**Args:**
  serialized_client_specs (bytes):  client specs to deserialize 



**Returns:**
  ClientSpecs:  deserialized client specs 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/specs.py#L40"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `serialize`

```python
serialize() → bytes
```

Serialize client specs into bytes. 



**Returns:**
  bytes:  serialized client specs 


