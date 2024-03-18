<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/library_compilation_result.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.compiler.library_compilation_result`
LibraryCompilationResult. 



---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/library_compilation_result.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LibraryCompilationResult`
LibraryCompilationResult holds the result of the library compilation. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/library_compilation_result.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(library_compilation_result: LibraryCompilationResult)
```

Wrap the native Cpp object. 



**Args:**
 
 - <b>`library_compilation_result`</b> (_LibraryCompilationResult):  object to wrap 



**Raises:**
 
 - <b>`TypeError`</b>:  if library_compilation_result is not of type _LibraryCompilationResult 




---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/library_compilation_result.py#L34"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `new`

```python
new(output_dir_path: str, func_name: str) → LibraryCompilationResult
```

Build a LibraryCompilationResult at output_dir_path, with func_name as entrypoint. 



**Args:**
 
 - <b>`output_dir_path`</b> (str):  path to the compilation artifacts 
 - <b>`func_name`</b> (str):  entrypoint function name 



**Raises:**
 
 - <b>`TypeError`</b>:  if output_dir_path is not of type str 
 - <b>`TypeError`</b>:  if func_name is not of type str 



**Returns:**
 LibraryCompilationResult 


