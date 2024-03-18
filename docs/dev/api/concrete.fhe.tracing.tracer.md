<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.tracing.tracer`
Declaration of `Tracer` class. 



---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Tracer`
Tracer class, to create computation graphs from python functions. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L171"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(computation: Node, input_tracers: List[ForwardRef('Tracer')])
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

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L620"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `astype`

```python
astype(
    dtype: Union[dtype[Any], NoneType, type[Any], _SupportsDType[dtype[Any]], str, tuple[Any, int], tuple[Any, Union[SupportsIndex, Sequence[SupportsIndex]]], list[Any], _DTypeDict, tuple[Any, Any], Type[ForwardRef('ScalarAnnotation')]]
) → Tracer
```

Trace numpy.ndarray.astype(dtype). 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L690"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `clip`

```python
clip(minimum: Any, maximum: Any) → Tracer
```

Trace numpy.ndarray.clip(). 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L699"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dot`

```python
dot(other: Any) → Tracer
```

Trace numpy.ndarray.dot(). 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L706"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `flatten`

```python
flatten() → Tracer
```

Trace numpy.ndarray.flatten(). 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L713"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `reshape`

```python
reshape(*newshape: Union[Any, Tuple[Any, ]]) → Tracer
```

Trace numpy.ndarray.reshape(newshape). 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L725"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `round`

```python
round(decimals: int = 0) → Tracer
```

Trace numpy.ndarray.round(). 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L191"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L36"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `trace`

```python
trace(
    function: Callable,
    parameters: Dict[str, ValueDescription],
    is_direct: bool = False
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

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L732"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `transpose`

```python
transpose(axes: Optional[Tuple[int, ]] = None) → Tracer
```

Trace numpy.ndarray.transpose(). 


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L928"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Annotation`
Base annotation for direct definition. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L171"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(computation: Node, input_tracers: List[ForwardRef('Tracer')])
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

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L620"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `astype`

```python
astype(
    dtype: Union[dtype[Any], NoneType, type[Any], _SupportsDType[dtype[Any]], str, tuple[Any, int], tuple[Any, Union[SupportsIndex, Sequence[SupportsIndex]]], list[Any], _DTypeDict, tuple[Any, Any], Type[ForwardRef('ScalarAnnotation')]]
) → Tracer
```

Trace numpy.ndarray.astype(dtype). 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L690"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `clip`

```python
clip(minimum: Any, maximum: Any) → Tracer
```

Trace numpy.ndarray.clip(). 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L699"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dot`

```python
dot(other: Any) → Tracer
```

Trace numpy.ndarray.dot(). 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L706"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `flatten`

```python
flatten() → Tracer
```

Trace numpy.ndarray.flatten(). 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L713"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `reshape`

```python
reshape(*newshape: Union[Any, Tuple[Any, ]]) → Tracer
```

Trace numpy.ndarray.reshape(newshape). 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L725"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `round`

```python
round(decimals: int = 0) → Tracer
```

Trace numpy.ndarray.round(). 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L191"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L36"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `trace`

```python
trace(
    function: Callable,
    parameters: Dict[str, ValueDescription],
    is_direct: bool = False
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

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L732"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `transpose`

```python
transpose(axes: Optional[Tuple[int, ]] = None) → Tracer
```

Trace numpy.ndarray.transpose(). 


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L934"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ScalarAnnotation`
Base scalar annotation for direct definition. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L171"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(computation: Node, input_tracers: List[ForwardRef('Tracer')])
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

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L620"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `astype`

```python
astype(
    dtype: Union[dtype[Any], NoneType, type[Any], _SupportsDType[dtype[Any]], str, tuple[Any, int], tuple[Any, Union[SupportsIndex, Sequence[SupportsIndex]]], list[Any], _DTypeDict, tuple[Any, Any], Type[ForwardRef('ScalarAnnotation')]]
) → Tracer
```

Trace numpy.ndarray.astype(dtype). 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L690"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `clip`

```python
clip(minimum: Any, maximum: Any) → Tracer
```

Trace numpy.ndarray.clip(). 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L699"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dot`

```python
dot(other: Any) → Tracer
```

Trace numpy.ndarray.dot(). 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L706"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `flatten`

```python
flatten() → Tracer
```

Trace numpy.ndarray.flatten(). 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L713"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `reshape`

```python
reshape(*newshape: Union[Any, Tuple[Any, ]]) → Tracer
```

Trace numpy.ndarray.reshape(newshape). 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L725"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `round`

```python
round(decimals: int = 0) → Tracer
```

Trace numpy.ndarray.round(). 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L191"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L36"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `trace`

```python
trace(
    function: Callable,
    parameters: Dict[str, ValueDescription],
    is_direct: bool = False
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

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L732"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `transpose`

```python
transpose(axes: Optional[Tuple[int, ]] = None) → Tracer
```

Trace numpy.ndarray.transpose(). 


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L942"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TensorAnnotation`
Base tensor annotation for direct definition. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L171"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(computation: Node, input_tracers: List[ForwardRef('Tracer')])
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

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L620"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `astype`

```python
astype(
    dtype: Union[dtype[Any], NoneType, type[Any], _SupportsDType[dtype[Any]], str, tuple[Any, int], tuple[Any, Union[SupportsIndex, Sequence[SupportsIndex]]], list[Any], _DTypeDict, tuple[Any, Any], Type[ForwardRef('ScalarAnnotation')]]
) → Tracer
```

Trace numpy.ndarray.astype(dtype). 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L690"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `clip`

```python
clip(minimum: Any, maximum: Any) → Tracer
```

Trace numpy.ndarray.clip(). 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L699"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dot`

```python
dot(other: Any) → Tracer
```

Trace numpy.ndarray.dot(). 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L706"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `flatten`

```python
flatten() → Tracer
```

Trace numpy.ndarray.flatten(). 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L713"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `reshape`

```python
reshape(*newshape: Union[Any, Tuple[Any, ]]) → Tracer
```

Trace numpy.ndarray.reshape(newshape). 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L725"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `round`

```python
round(decimals: int = 0) → Tracer
```

Trace numpy.ndarray.round(). 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L191"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L36"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `trace`

```python
trace(
    function: Callable,
    parameters: Dict[str, ValueDescription],
    is_direct: bool = False
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

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/tracing/tracer.py#L732"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `transpose`

```python
transpose(axes: Optional[Tuple[int, ]] = None) → Tracer
```

Trace numpy.ndarray.transpose(). 


