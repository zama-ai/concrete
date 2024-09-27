<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/tracing.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.tfhers.tracing`
Tracing of tfhers operations. 


---

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/tracing.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `to_native`

```python
to_native(value: Union[Tracer, TFHERSInteger]) → Union[Tracer, int, ndarray]
```

Convert a tfhers integer to the Concrete representation. 



**Args:**
 
 - <b>`value`</b> (Union[Tracer, TFHERSInteger]):  tfhers integer 



**Raises:**
 
 - <b>`TypeError`</b>:  wrong input type 



**Returns:**
 
 - <b>`Union[Tracer, int, ndarray]`</b>:  Tracer if the input is a tracer. int or ndarray otherwise. 


---

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/tracing.py#L46"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `from_native`

```python
from_native(
    value: Union[Tracer, int, ndarray],
    dtype_to: TFHERSIntegerType
) → Union[Tracer, int, ndarray]
```

Convert a Concrete integer to the tfhers representation. 

The returned value isn't wrapped in a TFHERSInteger, but should have its representation. 



**Args:**
 
 - <b>`value`</b> (Union[Tracer, int, ndarray]):  Concrete value to convert to tfhers 
 - <b>`dtype_to`</b> (TFHERSIntegerType):  tfhers integer type to convert to 



**Returns:**
 Union[Tracer, int, ndarray] 


