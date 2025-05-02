<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.tracing.tracer`
Declaration of `Tracer` class. 



---

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Tracer`
Tracer class, to create computation graphs from python functions. 

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L174"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(computation: Node, input_tracers: list['Tracer'])
```






---

#### <kbd>property</kbd> T

Trace numpy.ndarray.T. 

---

#### <kbd>property</kbd> ndim

Trace numpy.ndarray.ndim. 

---

#### <kbd>property</kbd> shape

Trace numpy.ndarray.shape. 

---

#### <kbd>property</kbd> size

Trace numpy.ndarray.size. 



---

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L639"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `astype`

```python
astype(
    dtype: Union[dtype[Any], NoneType, type[Any], _SupportsDType[dtype[Any]], str, tuple[Any, int], tuple[Any, Union[SupportsIndex, Sequence[SupportsIndex]]], list[Any], _DTypeDict, tuple[Any, Any], type['ScalarAnnotation']]
) → Tracer
```

Trace numpy.ndarray.astype(dtype). 

---

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L709"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `clip`

```python
clip(minimum: Any, maximum: Any) → Tracer
```

Trace numpy.ndarray.clip(). 

---

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L718"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dot`

```python
dot(other: Any) → Tracer
```

Trace numpy.ndarray.dot(). 

---

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L725"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `flatten`

```python
flatten() → Tracer
```

Trace numpy.ndarray.flatten(). 

---

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L732"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `reshape`

```python
reshape(*newshape: Union[Any, tuple[Any, ]]) → Tracer
```

Trace numpy.ndarray.reshape(newshape). 

---

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L744"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `round`

```python
round(decimals: int = 0) → Tracer
```

Trace numpy.ndarray.round(). 

---

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L194"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `sanitize`

```python
sanitize(value: Any) → Any
```

Try to create a tracer from a value. 



**Args:**
  value (Any):  value to use 



**Returns:**
  Any:  resulting tracer 

---

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L36"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `trace`

```python
trace(
    function: Callable,
    parameters: dict[str, ValueDescription],
    is_direct: bool = False,
    location: str = ''
) → Graph
```

Trace `function` and create the `Graph` that represents it. 



**Args:**
  function (Callable):  function to trace 

 parameters (Dict[str, ValueDescription]):  parameters of function to trace  e.g. parameter x is an EncryptedScalar holding a 7-bit UnsignedInteger 

 is_direct (bool, default = False):  whether the tracing is done on actual parameters or placeholders 



**Returns:**
  Graph:  computation graph corresponding to `function` 

---

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L751"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `transpose`

```python
transpose(axes: Optional[tuple[int, ]] = None) → Tracer
```

Trace numpy.ndarray.transpose(). 


---

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L1076"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Annotation`
Base annotation for direct definition. 

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L174"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(computation: Node, input_tracers: list['Tracer'])
```






---

#### <kbd>property</kbd> T

Trace numpy.ndarray.T. 

---

#### <kbd>property</kbd> ndim

Trace numpy.ndarray.ndim. 

---

#### <kbd>property</kbd> shape

Trace numpy.ndarray.shape. 

---

#### <kbd>property</kbd> size

Trace numpy.ndarray.size. 



---

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L639"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `astype`

```python
astype(
    dtype: Union[dtype[Any], NoneType, type[Any], _SupportsDType[dtype[Any]], str, tuple[Any, int], tuple[Any, Union[SupportsIndex, Sequence[SupportsIndex]]], list[Any], _DTypeDict, tuple[Any, Any], type['ScalarAnnotation']]
) → Tracer
```

Trace numpy.ndarray.astype(dtype). 

---

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L709"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `clip`

```python
clip(minimum: Any, maximum: Any) → Tracer
```

Trace numpy.ndarray.clip(). 

---

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L718"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dot`

```python
dot(other: Any) → Tracer
```

Trace numpy.ndarray.dot(). 

---

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L725"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `flatten`

```python
flatten() → Tracer
```

Trace numpy.ndarray.flatten(). 

---

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L732"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `reshape`

```python
reshape(*newshape: Union[Any, tuple[Any, ]]) → Tracer
```

Trace numpy.ndarray.reshape(newshape). 

---

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L744"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `round`

```python
round(decimals: int = 0) → Tracer
```

Trace numpy.ndarray.round(). 

---

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L194"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `sanitize`

```python
sanitize(value: Any) → Any
```

Try to create a tracer from a value. 



**Args:**
  value (Any):  value to use 



**Returns:**
  Any:  resulting tracer 

---

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L36"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `trace`

```python
trace(
    function: Callable,
    parameters: dict[str, ValueDescription],
    is_direct: bool = False,
    location: str = ''
) → Graph
```

Trace `function` and create the `Graph` that represents it. 



**Args:**
  function (Callable):  function to trace 

 parameters (Dict[str, ValueDescription]):  parameters of function to trace  e.g. parameter x is an EncryptedScalar holding a 7-bit UnsignedInteger 

 is_direct (bool, default = False):  whether the tracing is done on actual parameters or placeholders 



**Returns:**
  Graph:  computation graph corresponding to `function` 

---

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L751"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `transpose`

```python
transpose(axes: Optional[tuple[int, ]] = None) → Tracer
```

Trace numpy.ndarray.transpose(). 


---

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L1082"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ScalarAnnotation`
Base scalar annotation for direct definition. 

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L174"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(computation: Node, input_tracers: list['Tracer'])
```






---

#### <kbd>property</kbd> T

Trace numpy.ndarray.T. 

---

#### <kbd>property</kbd> ndim

Trace numpy.ndarray.ndim. 

---

#### <kbd>property</kbd> shape

Trace numpy.ndarray.shape. 

---

#### <kbd>property</kbd> size

Trace numpy.ndarray.size. 



---

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L639"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `astype`

```python
astype(
    dtype: Union[dtype[Any], NoneType, type[Any], _SupportsDType[dtype[Any]], str, tuple[Any, int], tuple[Any, Union[SupportsIndex, Sequence[SupportsIndex]]], list[Any], _DTypeDict, tuple[Any, Any], type['ScalarAnnotation']]
) → Tracer
```

Trace numpy.ndarray.astype(dtype). 

---

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L709"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `clip`

```python
clip(minimum: Any, maximum: Any) → Tracer
```

Trace numpy.ndarray.clip(). 

---

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L718"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dot`

```python
dot(other: Any) → Tracer
```

Trace numpy.ndarray.dot(). 

---

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L725"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `flatten`

```python
flatten() → Tracer
```

Trace numpy.ndarray.flatten(). 

---

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L732"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `reshape`

```python
reshape(*newshape: Union[Any, tuple[Any, ]]) → Tracer
```

Trace numpy.ndarray.reshape(newshape). 

---

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L744"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `round`

```python
round(decimals: int = 0) → Tracer
```

Trace numpy.ndarray.round(). 

---

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L194"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `sanitize`

```python
sanitize(value: Any) → Any
```

Try to create a tracer from a value. 



**Args:**
  value (Any):  value to use 



**Returns:**
  Any:  resulting tracer 

---

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L36"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `trace`

```python
trace(
    function: Callable,
    parameters: dict[str, ValueDescription],
    is_direct: bool = False,
    location: str = ''
) → Graph
```

Trace `function` and create the `Graph` that represents it. 



**Args:**
  function (Callable):  function to trace 

 parameters (Dict[str, ValueDescription]):  parameters of function to trace  e.g. parameter x is an EncryptedScalar holding a 7-bit UnsignedInteger 

 is_direct (bool, default = False):  whether the tracing is done on actual parameters or placeholders 



**Returns:**
  Graph:  computation graph corresponding to `function` 

---

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L751"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `transpose`

```python
transpose(axes: Optional[tuple[int, ]] = None) → Tracer
```

Trace numpy.ndarray.transpose(). 


---

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L1090"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TensorAnnotation`
Base tensor annotation for direct definition. 

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L174"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(computation: Node, input_tracers: list['Tracer'])
```






---

#### <kbd>property</kbd> T

Trace numpy.ndarray.T. 

---

#### <kbd>property</kbd> ndim

Trace numpy.ndarray.ndim. 

---

#### <kbd>property</kbd> shape

Trace numpy.ndarray.shape. 

---

#### <kbd>property</kbd> size

Trace numpy.ndarray.size. 



---

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L639"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `astype`

```python
astype(
    dtype: Union[dtype[Any], NoneType, type[Any], _SupportsDType[dtype[Any]], str, tuple[Any, int], tuple[Any, Union[SupportsIndex, Sequence[SupportsIndex]]], list[Any], _DTypeDict, tuple[Any, Any], type['ScalarAnnotation']]
) → Tracer
```

Trace numpy.ndarray.astype(dtype). 

---

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L709"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `clip`

```python
clip(minimum: Any, maximum: Any) → Tracer
```

Trace numpy.ndarray.clip(). 

---

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L718"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dot`

```python
dot(other: Any) → Tracer
```

Trace numpy.ndarray.dot(). 

---

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L725"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `flatten`

```python
flatten() → Tracer
```

Trace numpy.ndarray.flatten(). 

---

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L732"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `reshape`

```python
reshape(*newshape: Union[Any, tuple[Any, ]]) → Tracer
```

Trace numpy.ndarray.reshape(newshape). 

---

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L744"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `round`

```python
round(decimals: int = 0) → Tracer
```

Trace numpy.ndarray.round(). 

---

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L194"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `sanitize`

```python
sanitize(value: Any) → Any
```

Try to create a tracer from a value. 



**Args:**
  value (Any):  value to use 



**Returns:**
  Any:  resulting tracer 

---

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L36"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `trace`

```python
trace(
    function: Callable,
    parameters: dict[str, ValueDescription],
    is_direct: bool = False,
    location: str = ''
) → Graph
```

Trace `function` and create the `Graph` that represents it. 



**Args:**
  function (Callable):  function to trace 

 parameters (Dict[str, ValueDescription]):  parameters of function to trace  e.g. parameter x is an EncryptedScalar holding a 7-bit UnsignedInteger 

 is_direct (bool, default = False):  whether the tracing is done on actual parameters or placeholders 



**Returns:**
  Graph:  computation graph corresponding to `function` 

---

<a href="../../frontends/concrete-python/concrete/fhe/tracing/tracer.py#L751"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `transpose`

```python
transpose(axes: Optional[tuple[int, ]] = None) → Tracer
```

Trace numpy.ndarray.transpose(). 


