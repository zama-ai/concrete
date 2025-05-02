<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/extensions/multivariate.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.extensions.multivariate`
Declaration of `multivariate` extension. 


---

<a href="../../frontends/concrete-python/concrete/fhe/extensions/multivariate.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `multivariate`

```python
multivariate(
    function: Callable,
    outputs: Optional[BaseDataType, type[ScalarAnnotation]] = None
) → Callable
```

Wrap a multivariate function so that it is traced into a single generic node. 



**Args:**
  function (Callable[[Any, ...], Any]):  multivariate function to wrap 

 outputs (Optional[Union[BaseDataType, Type[ScalarAnnotation]]], default = None):  data type of the result  only required for direct circuits  ignored when compiling with inputsets 



**Returns:**
  Callable[[Union[Tracer, Any], ...], Union[Tracer, Any]]:  another multivariate function that can be called with Tracers as well 


