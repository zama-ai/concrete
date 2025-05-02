<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/compilation/artifacts.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.compilation.artifacts`
Declaration of `DebugArtifacts` class. 

**Global Variables**
---------------
- **TYPE_CHECKING**


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/artifacts.py#L23"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DebugManager`
A debug manager, allowing streamlined debugging. 

<a href="../../frontends/concrete-python/concrete/fhe/compilation/artifacts.py#L31"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(config: Configuration)
```








---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/artifacts.py#L186"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `debug_assigned_graph`

```python
debug_assigned_graph(name, function_graph)
```

Print assigned graphs if configuration tells so. 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/artifacts.py#L177"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `debug_bit_width_assignments`

```python
debug_bit_width_assignments(name, function_graph)
```

Print bitwidth assignments if configuration tells so. 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/artifacts.py#L168"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `debug_bit_width_constaints`

```python
debug_bit_width_constaints(name, function_graph)
```

Print bitwidth constraints if configuration tells so. 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/artifacts.py#L150"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `debug_computation_graph`

```python
debug_computation_graph(name, function_graph)
```

Print computation graph if configuration tells so. 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/artifacts.py#L195"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `debug_mlir`

```python
debug_mlir(mlir_str)
```

Print mlir if configuration tells so. 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/artifacts.py#L204"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `debug_statistics`

```python
debug_statistics(module)
```

Print statistics if configuration tells so. 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/artifacts.py#L42"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `debug_table`

```python
debug_table(title: str, activate: bool = True)
```

Return a context manager that prints a table around what is printed inside the scope. 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/artifacts.py#L106"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `show_assigned_graph`

```python
show_assigned_graph() → bool
```

Tell if the configuration involves showing assigned graph. 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/artifacts.py#L95"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `show_bit_width_assignments`

```python
show_bit_width_assignments() → bool
```

Tell if the configuration involves showing bitwidth assignments. 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/artifacts.py#L84"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `show_bit_width_constraints`

```python
show_bit_width_constraints() → bool
```

Tell if the configuration involves showing bitwidth constraints. 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/artifacts.py#L73"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `show_graph`

```python
show_graph() → bool
```

Tell if the configuration involves showing graph. 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/artifacts.py#L117"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `show_mlir`

```python
show_mlir() → bool
```

Tell if the configuration involves showing mlir. 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/artifacts.py#L128"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `show_optimizer`

```python
show_optimizer() → bool
```

Tell if the configuration involves showing optimizer. 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/artifacts.py#L139"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `show_statistics`

```python
show_statistics() → bool
```

Tell if the configuration involves showing statistics. 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/artifacts.py#L230"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FunctionDebugArtifacts`
An object containing debug artifacts for a certain function in an fhe module. 

<a href="../../frontends/concrete-python/concrete/fhe/compilation/artifacts.py#L240"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```








---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/artifacts.py#L274"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add_graph`

```python
add_graph(name: str, graph: Graph)
```

Add a representation of the function being compiled. 



**Args:**
  name (str):  name of the graph (e.g., initial, optimized, final) 

 graph (Graph):  a representation of the function being compiled 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/artifacts.py#L261"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add_parameter_encryption_status`

```python
add_parameter_encryption_status(name: str, encryption_status: str)
```

Add parameter encryption status of a parameter of the function being compiled. 



**Args:**
  name (str):  name of the parameter 

 encryption_status (str):  encryption status of the parameter 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/artifacts.py#L246"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add_source_code`

```python
add_source_code(function: Union[str, Callable])
```

Add source code of the function being compiled. 



**Args:**
  function (Union[str, Callable]):  either the source code of the function or the function itself 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/artifacts.py#L292"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ModuleDebugArtifacts`
An object containing debug artifacts for an fhe module. 

<a href="../../frontends/concrete-python/concrete/fhe/compilation/artifacts.py#L302"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    function_names: Optional[list[str]] = None,
    output_directory: Union[str, Path] = PosixPath('.artifacts')
)
```






---

#### <kbd>property</kbd> client_parameters

The client parameters associated with the execution runtime. 



---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/artifacts.py#L324"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add_execution_runtime`

```python
add_execution_runtime(execution_runtime: 'Lazy[ExecutionRt]')
```

Add the (lazy) execution runtime to get the client parameters if needed. 



**Args:**
  execution_runtime (Lazy[ExecutionRt]):  The lazily initialized execution runtime. 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/artifacts.py#L314"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add_mlir_to_compile`

```python
add_mlir_to_compile(mlir: str)
```

Add textual representation of the resulting MLIR. 



**Args:**
  mlir (str):  textual representation of the resulting MLIR 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/artifacts.py#L347"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `export`

```python
export()
```

Export the collected information to `self.output_directory`. 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/artifacts.py#L442"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DebugArtifacts`
DebugArtifacts class, to export information about the compilation process for single function. 

<a href="../../frontends/concrete-python/concrete/fhe/compilation/artifacts.py#L449"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(output_directory: Union[str, Path] = PosixPath('.artifacts'))
```






---

#### <kbd>property</kbd> mlir_to_compile

Return the mlir string. 

---

#### <kbd>property</kbd> output_directory

Return the directory to export artifacts to. 



---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/artifacts.py#L452"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `export`

```python
export()
```

Export the collected information to `self.output_directory`. 


