<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/artifacts.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.compilation.artifacts`
Declaration of `DebugArtifacts` class. 



---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/artifacts.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FunctionDebugArtifacts`
An object containing debug artifacts for a certain function in an fhe module. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/artifacts.py#L27"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```








---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/artifacts.py#L61"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add_graph`

```python
add_graph(name: str, graph: Graph)
```

Add a representation of the function being compiled. 



**Args:**
  name (str):  name of the graph (e.g., initial, optimized, final) 

 graph (Graph):  a representation of the function being compiled 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/artifacts.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add_parameter_encryption_status`

```python
add_parameter_encryption_status(name: str, encryption_status: str)
```

Add parameter encryption status of a parameter of the function being compiled. 



**Args:**
  name (str):  name of the parameter 

 encryption_status (str):  encryption status of the parameter 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/artifacts.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add_source_code`

```python
add_source_code(function: Union[str, Callable])
```

Add source code of the function being compiled. 



**Args:**
  function (Union[str, Callable]):  either the source code of the function or the function itself 


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/artifacts.py#L79"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ModuleDebugArtifacts`
An object containing debug artifacts for an fhe module. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/artifacts.py#L89"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    function_names: List[str],
    output_directory: Union[str, Path] = PosixPath('.artifacts')
)
```








---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/artifacts.py#L109"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add_client_parameters`

```python
add_client_parameters(client_parameters: bytes)
```

Add client parameters used. 



**Args:**
 
 - <b>`client_parameters`</b> (bytes):  client parameters 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/artifacts.py#L99"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add_mlir_to_compile`

```python
add_mlir_to_compile(mlir: str)
```

Add textual representation of the resulting MLIR. 



**Args:**
  mlir (str):  textual representation of the resulting MLIR 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/artifacts.py#L119"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `export`

```python
export()
```

Export the collected information to `self.output_directory`. 


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/artifacts.py#L212"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DebugArtifacts`
DebugArtifacts class, to export information about the compilation process for single function. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/artifacts.py#L219"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/artifacts.py#L273"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add_client_parameters`

```python
add_client_parameters(client_parameters: bytes)
```

Add client parameters used. 



**Args:**
 
 - <b>`client_parameters`</b> (bytes):  client parameters 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/artifacts.py#L248"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add_graph`

```python
add_graph(name: str, graph: Graph)
```

Add a representation of the function being compiled. 



**Args:**
  name (str):  name of the graph (e.g., initial, optimized, final) 

 graph (Graph):  a representation of the function being compiled 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/artifacts.py#L262"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add_mlir_to_compile`

```python
add_mlir_to_compile(mlir: str)
```

Add textual representation of the resulting MLIR. 



**Args:**
  mlir (str):  textual representation of the resulting MLIR 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/artifacts.py#L232"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add_parameter_encryption_status`

```python
add_parameter_encryption_status(name: str, encryption_status: str)
```

Add parameter encryption status of a parameter of the function being compiled. 



**Args:**
  name (str):  name of the parameter 

 encryption_status (str):  encryption status of the parameter 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/artifacts.py#L222"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add_source_code`

```python
add_source_code(function: Union[str, Callable])
```

Add source code of the function being compiled. 



**Args:**
  function (Union[str, Callable]):  either the source code of the function or the function itself 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/artifacts.py#L283"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `export`

```python
export()
```

Export the collected information to `self.output_directory`. 


