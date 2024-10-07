<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/compilation/keys.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.compilation.keys`
Declaration of `Keys` class. 



---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/keys.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Keys`
Keys class, to manage generate/reuse keys. 

Includes encryption keys as well as evaluation keys. Be careful when serializing/saving keys! 

<a href="../../frontends/concrete-python/concrete/fhe/compilation/keys.py#L32"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    client_specs: Optional[ClientSpecs],
    cache_directory: Optional[Path, str] = None
)
```






---

#### <kbd>property</kbd> are_generated

Get if the keys are already generated. 

---

#### <kbd>property</kbd> evaluation

Get only evaluation keys. 



---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/keys.py#L179"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `deserialize`

```python
deserialize(serialized_keys: bytes) → Keys
```

Deserialize keys from bytes. 



**Args:**
  serialized_keys (bytes):  previously serialized keys 



**Returns:**
  Keys:  deserialized keys 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/keys.py#L54"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `generate`

```python
generate(
    force: bool = False,
    seed: Optional[int] = None,
    encryption_seed: Optional[int] = None,
    initial_keys: Optional[Dict[int, LweSecretKey]] = None
)
```

Generate new keys. 



**Args:**
  force (bool, default = False):  whether to generate new keys even if keys are already generated/loaded 

 seed (Optional[int], default = None):  seed for private keys randomness 

 encryption_seed (Optional[int], default = None):  seed for encryption randomness 

 initial_keys (Optional[Dict[int, LweSecretKey]] = None):  initial keys to set before keygen 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/keys.py#L110"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load`

```python
load(location: Union[str, Path])
```

Load keys from a location. 



**Args:**
  location (Union[str, Path]):  location to load from 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/keys.py#L136"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load_if_exists_generate_and_save_otherwise`

```python
load_if_exists_generate_and_save_otherwise(
    location: Union[str, Path],
    seed: Optional[int] = None
)
```

Load keys from a location if they exist, else generate new keys and save to that location. 



**Args:**
  location (Union[str, Path]):  location to load from or save to 

 seed (Optional[int], default = None):  seed for randomness in case keys need to be generated 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/keys.py#L90"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(location: Union[str, Path])
```

Save keys to a location. 

Saved keys are not encrypted, so be careful how you store/transfer them! 



**Args:**
  location (Union[str, Path]):  location to save to 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/keys.py#L161"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `serialize`

```python
serialize() → bytes
```

Serialize keys into bytes. 

Serialized keys are not encrypted, so be careful how you store/transfer them! 



**Returns:**
  bytes:  serialized keys 


