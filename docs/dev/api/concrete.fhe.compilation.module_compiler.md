<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/module_compiler.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.compilation.module_compiler`
Declaration of `MultiCompiler` class. 



---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/module_compiler.py#L32"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FunctionDef`
An object representing the definition of a function as used in an fhe module. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/module_compiler.py#L46"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    function: Callable,
    parameter_encryption_statuses: Dict[str, Union[str, EncryptionStatus]]
)
```








---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/module_compiler.py#L137"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/module_compiler.py#L105"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `trace`

```python
trace(sample: Union[Any, Tuple[Any, ]])
```

Trace the function and fuse the resulting graph with a sample input. 



**Args:**
  sample (Union[Any, Tuple[Any, ...]]):  sample to use for tracing 


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/module_compiler.py#L245"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DebugManager`
A debug manager, allowing streamlined debugging. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/module_compiler.py#L253"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(config: Configuration)
```








---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/module_compiler.py#L408"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `debug_assigned_graph`

```python
debug_assigned_graph(name, function_graph)
```

Print assigned graphs if configuration tells so. 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/module_compiler.py#L399"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `debug_bit_width_assignments`

```python
debug_bit_width_assignments(name, function_graph)
```

Print bitwidth assignments if configuration tells so. 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/module_compiler.py#L390"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `debug_bit_width_constaints`

```python
debug_bit_width_constaints(name, function_graph)
```

Print bitwidth constraints if configuration tells so. 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/module_compiler.py#L372"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `debug_computation_graph`

```python
debug_computation_graph(name, function_graph)
```

Print computation graph if configuration tells so. 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/module_compiler.py#L417"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `debug_mlir`

```python
debug_mlir(mlir_str)
```

Print mlir if configuration tells so. 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/module_compiler.py#L426"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `debug_statistics`

```python
debug_statistics(module)
```

Print statistics if configuration tells so. 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/module_compiler.py#L264"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `debug_table`

```python
debug_table(title: str, activate: bool = True)
```

Return a context manager that prints a table around what is printed inside the scope. 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/module_compiler.py#L328"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `show_assigned_graph`

```python
show_assigned_graph() → bool
```

Tell if the configuration involves showing assigned graph. 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/module_compiler.py#L317"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `show_bit_width_assignments`

```python
show_bit_width_assignments() → bool
```

Tell if the configuration involves showing bitwidth assignments. 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/module_compiler.py#L306"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `show_bit_width_constraints`

```python
show_bit_width_constraints() → bool
```

Tell if the configuration involves showing bitwidth constraints. 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/module_compiler.py#L295"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `show_graph`

```python
show_graph() → bool
```

Tell if the configuration involves showing graph. 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/module_compiler.py#L339"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `show_mlir`

```python
show_mlir() → bool
```

Tell if the configuration involves showing mlir. 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/module_compiler.py#L350"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `show_optimizer`

```python
show_optimizer() → bool
```

Tell if the configuration involves showing optimizer. 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/module_compiler.py#L361"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `show_statistics`

```python
show_statistics() → bool
```

Tell if the configuration involves showing statistics. 


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/module_compiler.py#L452"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ModuleCompiler`
Compiler class for multiple functions, to glue the compilation pipeline. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/module_compiler.py#L461"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(functions: List[FunctionDef])
```








---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/module_compiler.py#L470"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(
    inputsets: Optional[Dict[str, Union[Iterable[Any], Iterable[Tuple[Any, ]]]]] = None,
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


