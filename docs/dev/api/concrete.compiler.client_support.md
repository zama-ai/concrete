<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/client_support.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.compiler.client_support`
Client support. 

**Global Variables**
---------------
- **ACCEPTED_INTS**
- **ACCEPTED_NUMPY_UINTS**
- **ACCEPTED_TYPES**


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/client_support.py#L24"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ClientSupport`
Client interface for doing key generation and encryption. 

It provides features that are needed on the client side: 
- Generation of public and private keys required for the encrypted computation 
- Encryption and preparation of public arguments, used later as input to the computation 
- Decryption of public result returned after execution 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/client_support.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(client_support: ClientSupport)
```

Wrap the native Cpp object. 



**Args:**
 
 - <b>`client_support`</b> (_ClientSupport):  object to wrap 



**Raises:**
 
 - <b>`TypeError`</b>:  if client_support is not of type _ClientSupport 




---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/client_support.py#L181"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `decrypt_result`

```python
decrypt_result(
    client_parameters: ClientParameters,
    keyset: KeySet,
    public_result: PublicResult,
    circuit_name: str
) → Union[int, ndarray]
```

Decrypt a public result using the keyset. 



**Args:**
 
 - <b>`client_parameters`</b> (ClientParameters):  client parameters for decryption 
 - <b>`keyset`</b> (KeySet):  keyset used for decryption 
 - <b>`public_result`</b> (PublicResult):  public result to decrypt 
 - <b>`circuit_name`</b> (str):  name of the circuit for which to decrypt 



**Raises:**
 
 - <b>`TypeError`</b>:  if keyset is not of type KeySet 
 - <b>`TypeError`</b>:  if public_result is not of type PublicResult 
 - <b>`TypeError`</b>:  if circuit_name is not of type str 
 - <b>`RuntimeError`</b>:  if the result is of an unknown type 



**Returns:**
 
 - <b>`Union[int, np.ndarray]`</b>:  plain result 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/client_support.py#L125"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `encrypt_arguments`

```python
encrypt_arguments(
    client_parameters: ClientParameters,
    keyset: KeySet,
    args: List[Union[int, ndarray]],
    circuit_name: str
) → PublicArguments
```

Prepare arguments for encrypted computation. 

Pack public arguments by encrypting the ones that requires encryption, and leaving the rest as plain. It also pack public materials (public keys) that are required during the computation. 



**Args:**
 
 - <b>`client_parameters`</b> (ClientParameters):  client parameters specification 
 - <b>`keyset`</b> (KeySet):  keyset used to encrypt arguments that require encryption 
 - <b>`args`</b> (List[Union[int, np.ndarray]]):  list of scalar or tensor arguments 
 - <b>`circuit_name`</b> (str):  the name of the circuit for which to encrypt 



**Raises:**
 
 - <b>`TypeError`</b>:  if client_parameters is not of type ClientParameters 
 - <b>`TypeError`</b>:  if keyset is not of type KeySet 
 - <b>`TypeError`</b>:  if circuit_name is not of type str 



**Returns:**
 
 - <b>`PublicArguments`</b>:  public arguments for execution 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/client_support.py#L60"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `key_set`

```python
key_set(
    client_parameters: ClientParameters,
    keyset_cache: Optional[KeySetCache] = None,
    secret_seed: Optional[int] = None,
    encryption_seed: Optional[int] = None,
    initial_lwe_secret_keys: Optional[Dict[int, LweSecretKey]] = None
) → KeySet
```

Generate a key set according to the client parameters. 

If the cache is set, and include equivalent keys as specified by the client parameters, the keyset is loaded, otherwise, a new keyset is generated and saved in the cache. If keygen is required, it will first initialize the secret keys provided, if any. 



**Args:**
 
 - <b>`client_parameters`</b> (ClientParameters):  client parameters specification 
 - <b>`keyset_cache`</b> (Optional[KeySetCache], optional):  keyset cache. Defaults to None. 
 - <b>`secret_seed`</b> (Optional[int]):  secret seed, must be a positive 128 bits integer 
 - <b>`encryption_seed`</b> (Optional[int]):  encryption seed, must be a positive 128 bits integer 
 - <b>`initial_lwe_secret_keys`</b> (Optional[Dict[int, LweSecretKey]]):  keys to init the keyset  with before keygen. It maps keyid to secret key 



**Raises:**
 
 - <b>`TypeError`</b>:  if client_parameters is not of type ClientParameters 
 - <b>`TypeError`</b>:  if keyset_cache is not of type KeySetCache 
 - <b>`AssertionError`</b>:  if seed components is not uint64 



**Returns:**
 
 - <b>`KeySet`</b>:  generated or loaded keyset 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/client_support.py#L49"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `new`

```python
new() → ClientSupport
```

Build a ClientSupport. 



**Returns:**
  ClientSupport 


