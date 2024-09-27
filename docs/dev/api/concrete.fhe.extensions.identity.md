<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/extensions/identity.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.extensions.identity`
Declaration of `identity` extension. 


---

<a href="../../frontends/concrete-python/concrete/fhe/extensions/identity.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `identity`

```python
identity(x: Union[Tracer, Any]) → Union[Tracer, Any]
```

Apply identity function to x. 

Bit-width of the input and the output can be different. 



**Args:**
  x (Union[Tracer, Any]):  input to identity 



**Returns:**
  Union[Tracer, Any]:  identity tracer if called with a tracer  deepcopy of the input otherwise 


---

<a href="../../frontends/concrete-python/concrete/fhe/extensions/identity.py#L40"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `refresh`

```python
refresh(x: Union[Tracer, Any]) → Union[Tracer, Any]
```

Refresh x. 

Refresh encryption noise, the output noise is usually smaller compared to the input noise. Bit-width of the input and the output can be different. 



**Args:**
  x (Union[Tracer, Any]):  input to identity 



**Returns:**
  Union[Tracer, Any]:  identity tracer if called with a tracer  deepcopy of the input otherwise 


