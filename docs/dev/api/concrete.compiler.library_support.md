<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/library_support.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.compiler.library_support`
LibrarySupport. 

Library support provides a way to compile an MLIR program into a library that can be later loaded to execute the compiled code. 

**Global Variables**
---------------
- **DEFAULT_OUTPUT_PATH**


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/library_support.py#L38"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LibrarySupport`
Support class for library compilation and execution. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/library_support.py#L41"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(library_support: LibrarySupport)
```

Wrap the native Cpp object. 



**Args:**
 
 - <b>`library_support`</b> (_LibrarySupport):  object to wrap 



**Raises:**
 
 - <b>`TypeError`</b>:  if library_support is not of type _LibrarySupport 


---

#### <kbd>property</kbd> output_dir_path

Path where to store compilation artifacts. 



---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/library_support.py#L132"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(
    mlir_program: Union[str, Module],
    options: CompilationOptions = <concrete.compiler.compilation_options.CompilationOptions object at ADDRESS>,
    compilation_context: Optional[CompilationContext] = None
) → LibraryCompilationResult
```

Compile an MLIR program using Concrete dialects into a library. 



**Args:**
 
 - <b>`mlir_program`</b> (Union[str, MlirModule]):  mlir program to compile (textual or in-memory) 
 - <b>`options`</b> (CompilationOptions):  compilation options 



**Raises:**
 
 - <b>`TypeError`</b>:  if mlir_program is not of type str or MlirModule 
 - <b>`TypeError`</b>:  if options is not of type CompilationOptions 



**Returns:**
 
 - <b>`LibraryCompilationResult`</b>:  the result of the library compilation 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/library_support.py#L347"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_program_info_path`

```python
get_program_info_path() → str
```

Get the path where the program info file is expected to be. 



**Returns:**
 
 - <b>`str`</b>:  path to the program info file 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/library_support.py#L339"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_shared_lib_path`

```python
get_shared_lib_path() → str
```

Get the path where the shared library is expected to be. 



**Returns:**
 
 - <b>`str`</b>:  path to the shared library 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/library_support.py#L194"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load_client_parameters`

```python
load_client_parameters(
    library_compilation_result: LibraryCompilationResult
) → ClientParameters
```

Load the client parameters from the library compilation result. 



**Args:**
 
 - <b>`library_compilation_result`</b> (LibraryCompilationResult):  compilation result of the library 



**Raises:**
 
 - <b>`TypeError`</b>:  if library_compilation_result is not of type LibraryCompilationResult 



**Returns:**
 
 - <b>`ClientParameters`</b>:  appropriate client parameters for the compiled library 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/library_support.py#L218"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load_compilation_feedback`

```python
load_compilation_feedback(
    compilation_result: LibraryCompilationResult
) → CompilationFeedback
```

Load the compilation feedback from the compilation result. 



**Args:**
 
 - <b>`compilation_result`</b> (LibraryCompilationResult):  result of the compilation 



**Raises:**
 
 - <b>`TypeError`</b>:  if compilation_result is not of type LibraryCompilationResult 



**Returns:**
 
 - <b>`CompilationFeedback`</b>:  the compilation feedback for the compiled program 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/library_support.py#L240"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load_server_lambda`

```python
load_server_lambda(
    library_compilation_result: LibraryCompilationResult,
    simulation: bool
) → LibraryLambda
```

Load the server lambda from the library compilation result. 



**Args:**
 
 - <b>`library_compilation_result`</b> (LibraryCompilationResult):  compilation result of the library 



**Raises:**
 
 - <b>`TypeError`</b>:  if library_compilation_result is not of type LibraryCompilationResult 



**Returns:**
 
 - <b>`LibraryLambda`</b>:  executable reference to the library 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/library_support.py#L69"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `new`

```python
new(
    output_path: str = '/Users/benoitchevalliermames/Documents/Zama/Git/concrete/tempdirectoryforapidocs/concrete-compiler_compilation_artifacts',
    runtime_library_path: Optional[str] = None,
    generateSharedLib: bool = True,
    generateStaticLib: bool = False,
    generateClientParameters: bool = True,
    generateCompilationFeedback: bool = True,
    generateCppHeader: bool = False
) → LibrarySupport
```

Build a LibrarySupport. 



**Args:**
 
 - <b>`output_path`</b> (str, optional):  path where to store compilation artifacts.  Defaults to DEFAULT_OUTPUT_PATH. 
 - <b>`runtime_library_path`</b> (Optional[str], optional):  path to the runtime library. Defaults to None. 
 - <b>`generateSharedLib`</b> (bool):  whether to emit shared library or not. Default to True. 
 - <b>`generateStaticLib`</b> (bool):  whether to emit static library or not. Default to False. 
 - <b>`generateClientParameters`</b> (bool):  whether to emit client parameters or not. Default to True. 
 - <b>`generateCppHeader`</b> (bool):  whether to emit cpp header or not. Default to False. 



**Raises:**
 
 - <b>`TypeError`</b>:  if output_path is not of type str 
 - <b>`TypeError`</b>:  if runtime_library_path is not of type str 
 - <b>`TypeError`</b>:  if one of the generation flags is not of type bool 



**Returns:**
 LibrarySupport 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/library_support.py#L181"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `reload`

```python
reload(func_name: str = 'main') → LibraryCompilationResult
```

Reload the library compilation result from the output_dir_path. 



**Args:**
 
 - <b>`func_name`</b>:  entrypoint function name 



**Returns:**
 
 - <b>`LibraryCompilationResult`</b>:  loaded library 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/library_support.py#L265"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `server_call`

```python
server_call(
    library_lambda: LibraryLambda,
    public_arguments: PublicArguments,
    evaluation_keys: EvaluationKeys
) → PublicResult
```

Call the library with public_arguments. 



**Args:**
 
 - <b>`library_lambda`</b> (LibraryLambda):  reference to the compiled library 
 - <b>`public_arguments`</b> (PublicArguments):  arguments to use for execution 
 - <b>`evaluation_keys`</b> (EvaluationKeys):  evaluation keys to use for execution 



**Raises:**
 
 - <b>`TypeError`</b>:  if library_lambda is not of type LibraryLambda 
 - <b>`TypeError`</b>:  if public_arguments is not of type PublicArguments 
 - <b>`TypeError`</b>:  if evaluation_keys is not of type EvaluationKeys 



**Returns:**
 
 - <b>`PublicResult`</b>:  result of the execution 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/library_support.py#L306"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `simulate`

```python
simulate(
    library_lambda: LibraryLambda,
    public_arguments: PublicArguments
) → PublicResult
```

Call the library with public_arguments in simulation mode. 



**Args:**
 
 - <b>`library_lambda`</b> (LibraryLambda):  reference to the compiled library 
 - <b>`public_arguments`</b> (PublicArguments):  arguments to use for execution 



**Raises:**
 
 - <b>`TypeError`</b>:  if library_lambda is not of type LibraryLambda 
 - <b>`TypeError`</b>:  if public_arguments is not of type PublicArguments 



**Returns:**
 
 - <b>`PublicResult`</b>:  result of the execution 


