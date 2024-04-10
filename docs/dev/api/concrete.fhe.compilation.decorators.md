<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/decorators.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.compilation.decorators`
Declaration of `circuit` and `compiler` decorators. 


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/decorators.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/decorators.py#L157"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compiler`

```python
compiler(parameters: Mapping[str, Union[str, EncryptionStatus]])
```

Provide an easy interface for the compilation of single-circuit programs. 



**Args:**
  parameters (Mapping[str, Union[str, EncryptionStatus]]):  encryption statuses of the parameters of the function to compile 


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/decorators.py#L172"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `module`

```python
module()
```

Provide an easy interface for the compilation of multi functions modules. 


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/decorators.py#L187"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `function`

```python
function(parameters: Dict[str, Union[str, EncryptionStatus]])
```

Provide an easy interface to define a function within an fhe module. 



**Args:**
  parameters (Mapping[str, Union[str, EncryptionStatus]]):  encryption statuses of the parameters of the function to compile 


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/decorators.py#L80"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Compilable`
Compilable class, to wrap a function and provide methods to trace and compile it. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/decorators.py#L88"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(function_: Callable, parameters)
```








---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/decorators.py#L126"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(
    inputset: Optional[Iterable[Any], Iterable[Tuple[Any, ]]] = None,
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

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/decorators.py#L96"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `trace`

```python
trace(
    inputset: Optional[Iterable[Any], Iterable[Tuple[Any, ]]] = None,
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


