<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/extensions/univariate.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.extensions.univariate`
Declaration of `univariate` function. 


---

<a href="../../frontends/concrete-python/concrete/fhe/extensions/univariate.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `univariate`

```python
univariate(
    function: Callable[[Any], Any],
    outputs: Optional[BaseDataType, type[ScalarAnnotation]] = None
) → Callable[[Union[Tracer, Any]], Union[Tracer, Any]]
```

Wrap a univariate function so that it is traced into a single generic node. 



**Args:**
  function (Callable[[Any], Any]):  univariate function to wrap 

 outputs (Optional[Union[BaseDataType, Type[ScalarAnnotation]]], default = None):  data type of the result, unused during compilation, required for direct definition 



**Returns:**
  Callable[[Union[Tracer, Any]], Union[Tracer, Any]]:  another univariate function that can be called with a Tracer as well 


