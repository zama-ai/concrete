<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/key_set_cache.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.compiler.key_set_cache`
KeySetCache. 

Cache for keys to avoid generating similar keys multiple times. 



---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/key_set_cache.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `KeySetCache`
KeySetCache is a cache for KeySet to avoid generating similar keys multiple times. 

Keys get cached and can be later used instead of generating a new keyset which can take a lot of time. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/key_set_cache.py#L24"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(keyset_cache: KeySetCache)
```

Wrap the native Cpp object. 



**Args:**
 
 - <b>`keyset_cache`</b> (_KeySetCache):  object to wrap 



**Raises:**
 
 - <b>`TypeError`</b>:  if keyset_cache is not of type _KeySetCache 




---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/key_set_cache.py#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `new`

```python
new(cache_path: str) → KeySetCache
```

Build a KeySetCache located at cache_path. 



**Args:**
 
 - <b>`cache_path`</b> (str):  path to the cache 



**Raises:**
 
 - <b>`TypeError`</b>:  if the path is not of type str. 



**Returns:**
 KeySetCache 


