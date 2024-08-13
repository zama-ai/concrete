<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/extensions/relu.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.extensions.relu`
Declaration of `relu` extension. 


---

<a href="../../frontends/concrete-python/concrete/fhe/extensions/relu.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `relu`

```python
relu(x: Union[Tracer, Any]) → Union[Tracer, Any]
```

Rectified linear unit extension. 

Computes:  x if x >= 0 else 0 



**Args:**
  x (Union[Tracer, Any]):  input to apply ReLU 



**Returns:**
  Union[Tracer, Any]:  Tracer that represent the operation during tracing  result of ReLU on `x` otherwise 


