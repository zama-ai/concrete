<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_feedback.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.compiler.compilation_feedback`
Compilation feedback. 


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_feedback.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `tag_from_location`

```python
tag_from_location(location)
```

Extract tag of the operation from its location. 


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_feedback.py#L43"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CircuitCompilationFeedback`
CircuitCompilationFeedback is a set of hint computed by the compiler engine for a circuit. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_feedback.py#L46"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(circuit_compilation_feedback: CircuitCompilationFeedback)
```

Wrap the native Cpp object. 



**Args:**
 
 - <b>`circuit_compilation_feeback`</b> (_CircuitCompilationFeedback):  object to wrap 



**Raises:**
 
 - <b>`TypeError`</b>:  if circuit_compilation_feedback is not of type _CircuitCompilationFeedback 




---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_feedback.py#L74"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `count`

```python
count(operations: Set[PrimitiveOperation]) → int
```

Count the amount of specified operations in the program. 



**Args:**
  operations (Set[PrimitiveOperation]):  set of operations used to filter the statistics 



**Returns:**
  int:  number of specified operations in the program 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_feedback.py#L93"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `count_per_parameter`

```python
count_per_parameter(
    operations: Set[PrimitiveOperation],
    key_types: Set[KeyType],
    client_parameters: ClientParameters
) → Dict[Parameter, int]
```

Count the amount of specified operations in the program and group by parameters. 



**Args:**
  operations (Set[PrimitiveOperation]):  set of operations used to filter the statistics 

 key_types (Set[KeyType]):  set of key types used to filter the statistics 

 client_parameters (ClientParameters):  client parameters required for grouping by parameters 



**Returns:**
  Dict[Parameter, int]:  number of specified operations per parameter in the program 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_feedback.py#L134"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `count_per_tag`

```python
count_per_tag(operations: Set[PrimitiveOperation]) → Dict[str, int]
```

Count the amount of specified operations in the program and group by tags. 



**Args:**
  operations (Set[PrimitiveOperation]):  set of operations used to filter the statistics 



**Returns:**
  Dict[str, int]:  number of specified operations per tag in the program 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_feedback.py#L167"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `count_per_tag_per_parameter`

```python
count_per_tag_per_parameter(
    operations: Set[PrimitiveOperation],
    key_types: Set[KeyType],
    client_parameters: ClientParameters
) → Dict[str, Dict[Parameter, int]]
```

Count the amount of specified operations in the program and group by tags and parameters. 



**Args:**
  operations (Set[PrimitiveOperation]):  set of operations used to filter the statistics 

 key_types (Set[KeyType]):  set of key types used to filter the statistics 

 client_parameters (ClientParameters):  client parameters required for grouping by parameters 



**Returns:**
  Dict[str, Dict[Parameter, int]]:  number of specified operations per tag per parameter in the program 


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_feedback.py#L220"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ProgramCompilationFeedback`
CompilationFeedback is a set of hint computed by the compiler engine. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_feedback.py#L223"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(program_compilation_feedback: ProgramCompilationFeedback)
```

Wrap the native Cpp object. 



**Args:**
 
 - <b>`compilation_feeback`</b> (_CompilationFeedback):  object to wrap 



**Raises:**
 
 - <b>`TypeError`</b>:  if program_compilation_feedback is not of type _CompilationFeedback 




---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_feedback.py#L257"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `circuit`

```python
circuit(circuit_name: str) → CircuitCompilationFeedback
```

Returns the feedback for the circuit circuit_name. 



**Args:**
  circuit_name (str):  the name of the circuit. 



**Returns:**
  CircuitCompilationFeedback:  the feedback for the circuit. 



**Raises:**
 
 - <b>`TypeError`</b>:  if the circuit_name is not a string 
 - <b>`ValueError`</b>:  if there is no circuit with name circuit_name 


