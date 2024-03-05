<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_context.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.compiler.compilation_context`
CompilationContext. 

CompilationContext holds the MLIR Context supposed to be used during IR generation. 



---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_context.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CompilationContext`
Support class for compilation context. 

CompilationContext is meant to outlive mlir_context(). Do not use the mlir_context after deleting the CompilationContext. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_context.py#L26"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(compilation_context: CompilationContext)
```

Wrap the native Cpp object. 



**Args:**
 
 - <b>`compilation_context`</b> (_CompilationContext):  object to wrap 



**Raises:**
 
 - <b>`TypeError`</b>:  if compilation_context is not of type _CompilationContext 




---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_context.py#L52"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `mlir_context`

```python
mlir_context() → Context
```

Get the MLIR context used by the compilation context. 

The Compilation Context should outlive the mlir_context. 



**Returns:**
 
 - <b>`MlirContext`</b>:  MLIR context of the compilation context 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_context.py#L43"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `new`

```python
new() → CompilationContext
```

Build a CompilationContext. 



**Returns:**
  CompilationContext 


