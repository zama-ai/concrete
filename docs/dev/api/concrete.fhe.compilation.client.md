<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/compilation/client.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.compilation.client`
Declaration of `Client` class. 



---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/client.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Client`
Client class, which can be used to manage keys, encrypt arguments and decrypt results. 

<a href="../../frontends/concrete-python/concrete/fhe/compilation/client.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    client_specs: ClientSpecs,
    keyset_cache_directory: Optional[Path, str] = None,
    is_simulated: bool = False
)
```






---

#### <kbd>property</kbd> evaluation_keys

Get evaluation keys for encrypted computation. 



**Returns:**
  EvaluationKeys  evaluation keys for encrypted computation 

---

#### <kbd>property</kbd> keys

Get the keys for the client. 

---

#### <kbd>property</kbd> specs

Get the client specs for the client. 



---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/client.py#L264"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `decrypt`

```python
decrypt(
    *results: Union[Value, tuple[Value, ]],
    function_name: Optional[str] = None
) → Union[int, ndarray, tuple[Union[int, ndarray, NoneType], ], NoneType]
```

Decrypt result(s) of evaluation. 



**Args:**
  *results (Union[Value, Tuple[Value, ...]]):  result(s) of evaluation  function_name (str):  name of the function to decrypt for 



**Returns:**
  Optional[Union[int, np.ndarray, Tuple[Optional[Union[int, np.ndarray]], ...]]]:  decrypted result(s) of evaluation 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/client.py#L156"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `encrypt`

```python
encrypt(
    *args: Optional[int, ndarray, list],
    function_name: Optional[str] = None
) → Union[Value, tuple[Optional[Value], ], NoneType]
```

Encrypt argument(s) to for evaluation. 



**Args:**
  *args (Optional[Union[int, np.ndarray, List]]):  argument(s) for evaluation  function_name (str):  name of the function to encrypt 



**Returns:**
  Optional[Union[Value, Tuple[Optional[Value], ...]]]:  encrypted argument(s) for evaluation 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/client.py#L124"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `keygen`

```python
keygen(
    force: bool = False,
    secret_seed: Optional[int] = None,
    encryption_seed: Optional[int] = None,
    initial_keys: Optional[dict[int, LweSecretKey]] = None
)
```

Generate keys required for homomorphic evaluation. 



**Args:**
  force (bool, default = False):  whether to generate new keys even if keys are already generated 

 secret_seed (Optional[int], default = None):  seed for private keys randomness 

 encryption_seed (Optional[int], default = None):  seed for encryption randomness 

 initial_keys (Optional[Dict[int, LweSecretKey]] = None):  initial keys to set before keygen 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/client.py#L63"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load`

```python
load(
    path: Union[str, Path],
    keyset_cache_directory: Optional[Path, str] = None,
    is_simulated: bool = False
) → Client
```

Load the client from the given path in zip format. 



**Args:**
  path (Union[str, Path]):  path to load the client from 

 keyset_cache_directory (Optional[Union[str, Path]], default = None):  keyset cache directory to use 

 is_simulated (bool, default = False):  should perform 



**Returns:**
  Client:  client loaded from the filesystem 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/client.py#L44"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(path: Union[str, Path])
```

Save the client into the given path in zip format. 



**Args:**
  path (Union[str, Path]):  path to save the client 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/client.py#L318"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `simulate_decrypt`

```python
simulate_decrypt(
    *results: Union[Value, tuple[Value, ]],
    function_name: Optional[str] = None
) → Union[int, ndarray, tuple[Union[int, ndarray, NoneType], ], NoneType]
```

Simulate decryption of result(s) of evaluation. 



**Args:**
  *results (Union[Value, Tuple[Value, ...]]):  result(s) of evaluation  function_name (str):  name of the function to decrypt for 



**Returns:**
  Optional[Union[int, np.ndarray, Tuple[Optional[Union[int, np.ndarray]], ...]]]:  decrypted result(s) of evaluation 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/client.py#L212"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `simulate_encrypt`

```python
simulate_encrypt(
    *args: Optional[int, ndarray, list],
    function_name: Optional[str] = None
) → Union[Value, tuple[Optional[Value], ], NoneType]
```

Simulate encryption of argument(s) for evaluation. 



**Args:**
  *args (Optional[Union[int, np.ndarray, List]]):  argument(s) for evaluation  function_name (str):  name of the function to encrypt 



**Returns:**
  Optional[Union[Value, Tuple[Optional[Value], ...]]]:  encrypted argument(s) for evaluation 


