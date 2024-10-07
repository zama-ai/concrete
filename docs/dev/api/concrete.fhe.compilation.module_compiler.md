<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.compilation.module_compiler`
Declaration of `MultiCompiler` class. 



---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FunctionDef`
An object representing the definition of a function as used in an fhe module. 

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    function: Callable,
    parameter_encryption_statuses: Dict[str, Union[str, EncryptionStatus]]
)
```






---

#### <kbd>property</kbd> name

Return the name of the function. 



---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L152"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `evaluate`

```python
evaluate(
    action: str,
    inputset: Optional[Iterable[Any], Iterable[Tuple[Any, ]]],
    configuration: Configuration,
    artifacts: FunctionDebugArtifacts
)
```

Trace, fuse, measure bounds, and update values in the resulting graph in one go. 



**Args:**
  action (str):  action being performed (e.g., "trace", "compile") 

 inputset (Optional[Union[Iterable[Any], Iterable[Tuple[Any, ...]]]]):  optional inputset to extend accumulated inputset before bounds measurement 

 configuration (Configuration):  configuration to be used 

 artifacts (FunctionDebugArtifacts):  artifact object to store informations in 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L114"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `trace`

```python
trace(
    sample: Union[Any, Tuple[Any, ]],
    artifacts: Optional[FunctionDebugArtifacts] = None
)
```

Trace the function and fuse the resulting graph with a sample input. 



**Args:**
  sample (Union[Any, Tuple[Any, ...]]):  sample to use for tracing 
 - <b>`artifacts`</b>:  Optiona[FunctionDebugArtifacts]:  the object to store artifacts in 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L330"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ModuleCompiler`
Compiler class for multiple functions, to glue the compilation pipeline. 

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L340"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(functions: List[FunctionDef], composition: CompositionPolicy)
```








---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L356"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(
    inputsets: Optional[Dict[str, Optional[Iterable[Any], Iterable[Tuple[Any, ]]]]] = None,
    configuration: Optional[Configuration] = None,
    module_artifacts: Optional[ModuleDebugArtifacts] = None,
    **kwargs
) → FheModule
```

Compile the module using an ensemble of inputsets. 



**Args:**
  inputsets (Optional[Dict[str, Union[Iterable[Any], Iterable[Tuple[Any, ...]]]]]):  optional inputsets to extend accumulated inputsets before bounds measurement 

 configuration(Optional[Configuration], default = None):  configuration to use 

 artifacts (Optional[ModuleDebugArtifacts], default = None):  artifacts to store information about the process 

 kwargs (Dict[str, Any]):  configuration options to overwrite 



**Returns:**
  FheModule:  compiled module 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L349"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `wire_pipeline`

```python
wire_pipeline(inputset: Union[Iterable[Any], Iterable[Tuple[Any, ]]])
```

Return a context manager that traces wires automatically. 


