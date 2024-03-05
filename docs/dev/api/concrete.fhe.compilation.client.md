<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/client.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.compilation.client`
Declaration of `Client` class. 



---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/client.py#L23"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Client`
Client class, which can be used to manage keys, encrypt arguments and decrypt results. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/client.py#L31"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    client_specs: ClientSpecs,
    keyset_cache_directory: Optional[Path, str] = None
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

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/client.py#L155"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `decrypt`

```python
decrypt(
    *results: Union[Value, Tuple[Value, ]]
) → Union[int, ndarray, Tuple[Union[int, ndarray, NoneType], ], NoneType]
```

Decrypt result(s) of evaluation. 



**Args:**
  *results (Union[Value, Tuple[Value, ...]]):  result(s) of evaluation 



**Returns:**
  Optional[Union[int, np.ndarray, Tuple[Optional[Union[int, np.ndarray]], ...]]]:  decrypted result(s) of evaluation 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/client.py#L120"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `encrypt`

```python
encrypt(
    *args: Optional[int, ndarray, List]
) → Union[Value, Tuple[Optional[Value], ], NoneType]
```

Encrypt argument(s) to for evaluation. 



**Args:**
  *args (Optional[Union[int, np.ndarray, List]]):  argument(s) for evaluation 



**Returns:**
  Optional[Union[Value, Tuple[Optional[Value], ...]]]:  encrypted argument(s) for evaluation 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/client.py#L101"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `keygen`

```python
keygen(
    force: bool = False,
    seed: Optional[int] = None,
    encryption_seed: Optional[int] = None
)
```

Generate keys required for homomorphic evaluation. 



**Args:**
  force (bool, default = False):  whether to generate new keys even if keys are already generated 

 seed (Optional[int], default = None):  seed for private keys randomness 

 encryption_seed (Optional[int], default = None):  seed for encryption randomness 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/client.py#L58"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load`

```python
load(
    path: Union[str, Path],
    keyset_cache_directory: Optional[Path, str] = None
) → Client
```

Load the client from the given path in zip format. 



**Args:**
  path (Union[str, Path]):  path to load the client from 

 keyset_cache_directory (Optional[Union[str, Path]], default = None):  keyset cache directory to use 



**Returns:**
  Client:  client loaded from the filesystem 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/client.py#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(path: Union[str, Path])
```

Save the client into the given path in zip format. 



**Args:**
  path (Union[str, Path]):  path to save the client 


