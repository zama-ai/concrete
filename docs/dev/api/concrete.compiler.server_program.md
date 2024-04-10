<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/server_program.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.compiler.server_program`
ServerProgram. 



---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/server_program.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ServerProgram`
ServerProgram references compiled circuit objects. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/server_program.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(server_program: ServerProgram)
```

Wrap the native Cpp object. 



**Args:**
 
 - <b>`server_program`</b> (_ServerProgram):  object to wrap 



**Raises:**
 
 - <b>`TypeError`</b>:  if server_program is not of type _ServerProgram 




---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/server_program.py#L65"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_server_circuit`

```python
get_server_circuit(circuit_name: str) → ServerCircuit
```

Returns a given circuit if it is part of the program. 



**Args:**
 
 - <b>`circuit_name`</b> (str):  name of the circuit to retrieve. 



**Raises:**
 
 - <b>`TypeError`</b>:  if circuit_name is not of type str 
 - <b>`RuntimeError`</b>:  if the circuit is not part of the program 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/server_program.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load`

```python
load(library_support: LibrarySupport, simulation: bool) → ServerProgram
```

Loads the server program from a library support. 



**Args:**
 
 - <b>`library_support`</b> (LibrarySupport):  library support 
 - <b>`simulation`</b> (bool):  use simulation for execution 



**Raises:**
 
 - <b>`TypeError`</b>:  if library_support is not of type LibrarySupport, or if simulation is not of type bool 



**Returns:**
 
 - <b>`ServerProgram`</b>:  A server program object containing references to circuits for calls. 


