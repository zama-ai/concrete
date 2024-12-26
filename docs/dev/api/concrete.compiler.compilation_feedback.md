<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_feedback.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.compiler.compilation_feedback`
Compilation feedback. 


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_feedback.py#L23"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `tag_from_location`

```python
tag_from_location(location)
```

Extract tag of the operation from its location. 


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_feedback.py#L38"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MoreCircuitCompilationFeedback`
Helper class for compilation feedback. 




---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_feedback.py#L43"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `count`

```python
count(
    circuit_feedback: CircuitCompilationFeedback,
    operations: Set[PrimitiveOperation]
) → int
```

Count the amount of specified operations in the program. 



**Args:**
  operations (Set[PrimitiveOperation]):  set of operations used to filter the statistics 



**Returns:**
  int:  number of specified operations in the program 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_feedback.py#L67"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `count_per_parameter`

```python
count_per_parameter(
    circuit_feedback: CircuitCompilationFeedback,
    operations: Set[PrimitiveOperation],
    key_types: Set[KeyType],
    program_info: ProgramInfo
) → Dict[ForwardRef('Parameter'), int]
```

Count the amount of specified operations in the program and group by parameters. 



**Args:**
  operations (Set[PrimitiveOperation]):  set of operations used to filter the statistics 

 key_types (Set[KeyType]):  set of key types used to filter the statistics 

 program_info (ProgramInfo):  program info required for grouping by parameters 



**Returns:**
  Dict[Parameter, int]:  number of specified operations per parameter in the program 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_feedback.py#L124"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `count_per_tag`

```python
count_per_tag(
    circuit_feedback: CircuitCompilationFeedback,
    operations: Set[PrimitiveOperation]
) → Dict[str, int]
```

Count the amount of specified operations in the program and group by tags. 



**Args:**
  operations (Set[PrimitiveOperation]):  set of operations used to filter the statistics 



**Returns:**
  Dict[str, int]:  number of specified operations per tag in the program 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_feedback.py#L162"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `count_per_tag_per_parameter`

```python
count_per_tag_per_parameter(
    circuit_feedback: CircuitCompilationFeedback,
    operations: Set[PrimitiveOperation],
    key_types: Set[KeyType],
    program_info: ProgramInfo
) → Dict[str, Dict[ForwardRef('Parameter'), int]]
```

Count the amount of specified operations in the program and group by tags and parameters. 



**Args:**
  operations (Set[PrimitiveOperation]):  set of operations used to filter the statistics 

 key_types (Set[KeyType]):  set of key types used to filter the statistics 

 program_info (ProgramInfo):  program info required for grouping by parameters 



**Returns:**
  Dict[str, Dict[Parameter, int]]:  number of specified operations per tag per parameter in the program 


