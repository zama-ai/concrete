<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/compilation/keys.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.compilation.keys`
Declaration of `Keys` class. 



---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/keys.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Keys`
Keys class, to manage generate/reuse keys. 

Includes encryption keys as well as evaluation keys. Be careful when serializing/saving keys! 

<a href="../../frontends/concrete-python/concrete/fhe/compilation/keys.py#L31"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    specs: Optional[ClientSpecs],
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

#### <kbd>property</kbd> specs

Return the associated client specs if any. 



---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/keys.py#L220"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `deserialize`

```python
deserialize(serialized_keys: Union[Path, bytes]) → Keys
```

Deserialize keys from file or buffer. 

Prefer using a Path instead of bytes in case of big Keys. It reduces memory usage. 



**Args:**
  serialized_keys (Union[Path, bytes]):  previously serialized keys (either Path or buffer) 



**Returns:**
  Keys:  deserialized keys 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/keys.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `generate`

```python
generate(
    force: bool = False,
    secret_seed: Optional[int] = None,
    encryption_seed: Optional[int] = None,
    initial_keys: Optional[dict[int, LweSecretKey]] = None
)
```

Generate new keys. 



**Args:**
  force (bool, default = False):  whether to generate new keys even if keys are already generated/loaded 

 secret_seed (Optional[int], default = None):  seed for private keys randomness 

 encryption_seed (Optional[int], default = None):  seed for encryption randomness 

 initial_keys (Optional[Dict[int, LweSecretKey]] = None):  initial keys to set before keygen 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/keys.py#L120"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load`

```python
load(location: Union[str, Path])
```

Load keys from a location. 



**Args:**
  location (Union[str, Path]):  location to load from 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/keys.py#L144"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load_from_bytes`

```python
load_from_bytes(serialized_keys: bytes)
```

Load keys from bytes. 



**Args:**
 
 - <b>`serialized_keys`</b> (bytes):  serialized keys to load from 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/keys.py#L158"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load_if_exists_generate_and_save_otherwise`

```python
load_if_exists_generate_and_save_otherwise(
    location: Union[str, Path],
    secret_seed: Optional[int] = None
)
```

Load keys from a location if they exist, else generate new keys and save to that location. 



**Args:**
  location (Union[str, Path]):  location to load from or save to 

 secret_seed (Optional[int], default = None):  seed for randomness in case keys need to be generated 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/keys.py#L100"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(location: Union[str, Path])
```

Save keys to a location. 

Saved keys are not encrypted, so be careful how you store/transfer them! 



**Args:**
  location (Union[str, Path]):  location to save to 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/keys.py#L183"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `serialize`

```python
serialize() → bytes
```

Serialize keys into bytes. 

Serialized keys are not encrypted, so be careful how you store/transfer them! `serialize_to_file` is supposed to be more performant as it avoid copying the buffer between the Compiler and the Frontend. 



**Returns:**
  bytes:  serialized keys 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/keys.py#L203"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `serialize_to_file`

```python
serialize_to_file(path: Path)
```

Serialize keys into a file. 

Serialized keys are not encrypted, so be careful how you store/transfer them! This is supposed to be more performant than `serialize` as it avoid copying the buffer between the Compiler and the Frontend. 



**Args:**
 
 - <b>`path`</b> (Path):  where to save serialized keys 


