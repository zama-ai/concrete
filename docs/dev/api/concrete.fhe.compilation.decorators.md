<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/compilation/decorators.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.compilation.decorators`
Declaration of `circuit` and `compiler` decorators. 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/decorators.py#L23"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `circuit`

```python
circuit(
    parameters: Mapping[str, Union[str, EncryptionStatus]],
    configuration: Optional[Configuration] = None,
    artifacts: Optional[DebugArtifacts] = None,
    **kwargs
)
```

Provide a direct interface for compilation of single circuit programs. 



**Args:**
  parameters (Mapping[str, Union[str, EncryptionStatus]]):  encryption statuses of the parameters of the function to compile 

 configuration(Optional[Configuration], default = None):  configuration to use 

 artifacts (Optional[DebugArtifacts], default = None):  artifacts to store information about the process 

 kwargs (Dict[str, Any]):  configuration options to overwrite 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/decorators.py#L169"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compiler`

```python
compiler(parameters: Mapping[str, Union[str, EncryptionStatus]])
```

Provide an easy interface for the compilation of single-circuit programs. 



**Args:**
  parameters (Mapping[str, Union[str, EncryptionStatus]]):  encryption statuses of the parameters of the function to compile 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/decorators.py#L184"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `module`

```python
module()
```

Provide an easy interface for the compilation of multi functions modules. 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/decorators.py#L201"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `function`

```python
function(parameters: dict[str, Union[str, EncryptionStatus]])
```

Provide an easy interface to define a function within an fhe module. 



**Args:**
  parameters (Mapping[str, Union[str, EncryptionStatus]]):  encryption statuses of the parameters of the function to compile 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/decorators.py#L83"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Compilable`
Compilable class, to wrap a function and provide methods to trace and compile it. 

<a href="../../frontends/concrete-python/concrete/fhe/compilation/decorators.py#L91"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(function_: Callable, parameters)
```








---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/decorators.py#L129"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(
    inputset: Optional[Iterable[Any], Iterable[tuple[Any, ]]] = None,
    configuration: Optional[Configuration] = None,
    artifacts: Optional[DebugArtifacts] = None,
    **kwargs
) → Circuit
```

Compile the function into a circuit. 



**Args:**
  inputset (Optional[Union[Iterable[Any], Iterable[Tuple[Any, ...]]]]):  optional inputset to extend accumulated inputset before bounds measurement 

 configuration(Optional[Configuration], default = None):  configuration to use 

 artifacts (Optional[DebugArtifacts], default = None):  artifacts to store information about the process 

 kwargs (Dict[str, Any]):  configuration options to overwrite 



**Returns:**
  Circuit:  compiled circuit 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/decorators.py#L161"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `reset`

```python
reset()
```

Reset the compilable so that another compilation with another inputset can be performed. 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/decorators.py#L99"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `trace`

```python
trace(
    inputset: Optional[Iterable[Any], Iterable[tuple[Any, ]]] = None,
    configuration: Optional[Configuration] = None,
    artifacts: Optional[DebugArtifacts] = None,
    **kwargs
) → Graph
```

Trace the function into computation graph. 



**Args:**
  inputset (Optional[Union[Iterable[Any], Iterable[Tuple[Any, ...]]]]):  optional inputset to extend accumulated inputset before bounds measurement 

 configuration(Optional[Configuration], default = None):  configuration to use 

 artifacts (Optional[DebugArtifacts], default = None):  artifacts to store information about the process 

 kwargs (Dict[str, Any]):  configuration options to overwrite 



**Returns:**
  Graph:  computation graph representing the function prior to MLIR conversion 


