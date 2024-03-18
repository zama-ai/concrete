<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_feedback.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.compiler.compilation_feedback`
Compilation feedback. 


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_feedback.py#L27"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `tag_from_location`

```python
tag_from_location(location)
```

Extract tag of the operation from its location. 


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_feedback.py#L42"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CompilationFeedback`
CompilationFeedback is a set of hint computed by the compiler engine. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_feedback.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(compilation_feedback: CompilationFeedback)
```

Wrap the native Cpp object. 



**Args:**
 
 - <b>`compilation_feeback`</b> (_CompilationFeedback):  object to wrap 



**Raises:**
 
 - <b>`TypeError`</b>:  if compilation_feedback is not of type _CompilationFeedback 




---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_feedback.py#L75"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_feedback.py#L94"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_feedback.py#L135"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_feedback.py#L168"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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


