<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/compilation/compiler.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.compilation.compiler`
Declaration of `Compiler` class. 



---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/compiler.py#L26"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Compiler`
Compiler class, to glue the compilation pipeline. 

<a href="../../frontends/concrete-python/concrete/fhe/compilation/compiler.py#L86"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    function: Callable,
    parameter_encryption_statuses: dict[str, Union[str, EncryptionStatus]],
    composition: Optional[NotComposable, AllComposable] = None
)
```








---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/compiler.py#L34"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `assemble`

```python
assemble(
    function: Callable,
    parameter_values: dict[str, ValueDescription],
    configuration: Optional[Configuration] = None,
    artifacts: Optional[DebugArtifacts] = None,
    **kwargs
) → Circuit
```

Assemble a circuit from the raw parameter values, used in direct circuit definition. 



**Args:**
  function (Callable):  function to convert to a circuit 

 parameter_values (Dict[str, ValueDescription]):  parameter values of the function 

 configuration(Optional[Configuration], default = None):  configuration to use 

 artifacts (Optional[DebugArtifacts], default = None):  artifacts to store information about the process 

 kwargs (Dict[str, Any]):  configuration options to overwrite 



**Returns:**
  Circuit:  assembled circuit 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/compiler.py#L165"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(
    inputset: Optional[Iterable[Any], Iterable[tuple[Any, ]]] = None,
    configuration: Optional[Configuration] = None,
    artifacts: Optional[DebugArtifacts] = None,
    **kwargs
) → Circuit
```

Compile the function using an inputset. 



**Args:**
  inputset (Optional[Union[Iterable[Any], Iterable[Tuple[Any, ...]]]]):  optional inputset to extend accumulated inputset before bounds measurement 

 configuration(Optional[Configuration], default = None):  configuration to use 

 artifacts (Optional[DebugArtifacts], default = None):  artifacts to store information about the process 

 kwargs (Dict[str, Any]):  configuration options to overwrite 



**Returns:**
  Circuit:  compiled circuit 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/compiler.py#L211"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `reset`

```python
reset()
```

Reset the compiler so that another compilation with another inputset can be performed. 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/compiler.py#L119"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `trace`

```python
trace(
    inputset: Optional[Iterable[Any], Iterable[tuple[Any, ]]] = None,
    configuration: Optional[Configuration] = None,
    artifacts: Optional[DebugArtifacts] = None,
    **kwargs
) → Graph
```

Trace the function using an inputset. 



**Args:**
  inputset (Optional[Union[Iterable[Any], Iterable[Tuple[Any, ...]]]]):  optional inputset to extend accumulated inputset before bounds measurement 

 configuration(Optional[Configuration], default = None):  configuration to use 

 artifacts (Optional[DebugArtifacts], default = None):  artifacts to store information about the process 

 kwargs (Dict[str, Any]):  configuration options to overwrite 



**Returns:**
  Graph:  computation graph representing the function prior to MLIR conversion 


