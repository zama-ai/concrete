<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.compilation.module_compiler`
Declaration of `MultiCompiler` class. 



---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L46"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FunctionDef`
An object representing the definition of a function as used in an fhe module. 

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L58"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    function: Callable,
    parameter_encryption_statuses: Dict[str, Union[str, EncryptionStatus]]
)
```








---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L153"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L115"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L252"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `NotComposable`
Composition policy that does not allow the forwarding of any output to any input. 




---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L257"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_rules_iter`

```python
get_rules_iter(_funcs: List[FunctionDef]) → Iterable[CompositionRule]
```

Return an iterator over composition rules. 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L264"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AllComposable`
Composition policy that allows to forward any output of the module to any of its input. 




---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L269"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_rules_iter`

```python
get_rules_iter(funcs: List[Graph]) → Iterable[CompositionRule]
```

Return an iterator over composition rules. 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L288"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `WireOutput`
A protocol for wire outputs. 




---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L294"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_outputs_iter`

```python
get_outputs_iter() → Iterable[CompositionClause]
```

Return an iterator over the possible outputs of the wire output. 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L300"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `WireInput`
A protocol for wire inputs. 




---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L306"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_inputs_iter`

```python
get_inputs_iter() → Iterable[CompositionClause]
```

Return an iterator over the possible inputs of the wire input. 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L312"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Output`
The output of a given function of a module. 




---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L320"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_outputs_iter`

```python
get_outputs_iter() → Iterable[CompositionClause]
```

Return an iterator over the possible outputs of the wire output. 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L327"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AllOutputs`
All the outputs of a given function of a module. 




---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L334"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_outputs_iter`

```python
get_outputs_iter() → Iterable[CompositionClause]
```

Return an iterator over the possible outputs of the wire output. 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L345"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Input`
The input of a given function of a module. 




---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L353"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_inputs_iter`

```python
get_inputs_iter() → Iterable[CompositionClause]
```

Return an iterator over the possible inputs of the wire input. 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L360"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AllInputs`
All the inputs of a given function of a module. 




---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L367"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_inputs_iter`

```python
get_inputs_iter() → Iterable[CompositionClause]
```

Return an iterator over the possible inputs of the wire input. 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L378"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Wire`
A forwarding rule between an output and an input. 




---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L386"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_rules_iter`

```python
get_rules_iter(_) → Iterable[CompositionRule]
```

Return an iterator over composition rules. 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L396"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Wired`
Composition policy which allows the forwarding of certain outputs to certain inputs. 




---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L403"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_rules_iter`

```python
get_rules_iter(_) → Iterable[CompositionRule]
```

Return an iterator over composition rules. 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L410"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DebugManager`
A debug manager, allowing streamlined debugging. 

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L418"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(config: Configuration)
```








---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L573"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `debug_assigned_graph`

```python
debug_assigned_graph(name, function_graph)
```

Print assigned graphs if configuration tells so. 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L564"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `debug_bit_width_assignments`

```python
debug_bit_width_assignments(name, function_graph)
```

Print bitwidth assignments if configuration tells so. 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L555"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `debug_bit_width_constaints`

```python
debug_bit_width_constaints(name, function_graph)
```

Print bitwidth constraints if configuration tells so. 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L537"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `debug_computation_graph`

```python
debug_computation_graph(name, function_graph)
```

Print computation graph if configuration tells so. 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L582"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `debug_mlir`

```python
debug_mlir(mlir_str)
```

Print mlir if configuration tells so. 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L591"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `debug_statistics`

```python
debug_statistics(module)
```

Print statistics if configuration tells so. 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L429"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `debug_table`

```python
debug_table(title: str, activate: bool = True)
```

Return a context manager that prints a table around what is printed inside the scope. 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L493"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `show_assigned_graph`

```python
show_assigned_graph() → bool
```

Tell if the configuration involves showing assigned graph. 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L482"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `show_bit_width_assignments`

```python
show_bit_width_assignments() → bool
```

Tell if the configuration involves showing bitwidth assignments. 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L471"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `show_bit_width_constraints`

```python
show_bit_width_constraints() → bool
```

Tell if the configuration involves showing bitwidth constraints. 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L460"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `show_graph`

```python
show_graph() → bool
```

Tell if the configuration involves showing graph. 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L504"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `show_mlir`

```python
show_mlir() → bool
```

Tell if the configuration involves showing mlir. 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L515"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `show_optimizer`

```python
show_optimizer() → bool
```

Tell if the configuration involves showing optimizer. 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L526"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `show_statistics`

```python
show_statistics() → bool
```

Tell if the configuration involves showing statistics. 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L617"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ModuleCompiler`
Compiler class for multiple functions, to glue the compilation pipeline. 

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L627"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(functions: List[FunctionDef], composition: CompositionPolicy)
```








---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/module_compiler.py#L636"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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


