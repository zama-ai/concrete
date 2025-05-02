<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.compilation.wiring`
Declaration of wiring related class. 

**Global Variables**
---------------
- **TYPE_CHECKING**


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `NotComposable`
Composition policy that does not allow the forwarding of any output to any input. 




---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L23"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_rules_iter`

```python
get_rules_iter(_funcs: list['FunctionDef']) → Iterable[CompositionRule]
```

Return an iterator over composition rules. 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L30"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AllComposable`
Composition policy that allows to forward any output of the module to any of its input. 




---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_rules_iter`

```python
get_rules_iter(funcs: list[Graph]) → Iterable[CompositionRule]
```

Return an iterator over composition rules. 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L53"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `WireOutput`
A protocol for wire outputs. 




---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L59"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_outputs_iter`

```python
get_outputs_iter() → Iterable[CompositionClause]
```

Return an iterator over the possible outputs of the wire output. 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L65"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `WireInput`
A protocol for wire inputs. 




---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L71"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_inputs_iter`

```python
get_inputs_iter() → Iterable[CompositionClause]
```

Return an iterator over the possible inputs of the wire input. 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L77"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Output`
The output of a given function of a module. 




---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L85"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_outputs_iter`

```python
get_outputs_iter() → Iterable[CompositionClause]
```

Return an iterator over the possible outputs of the wire output. 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L92"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AllOutputs`
All the encrypted outputs of a given function of a module. 




---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L99"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_outputs_iter`

```python
get_outputs_iter() → Iterable[CompositionClause]
```

Return an iterator over the possible outputs of the wire output. 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L111"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Input`
The input of a given function of a module. 




---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L119"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_inputs_iter`

```python
get_inputs_iter() → Iterable[CompositionClause]
```

Return an iterator over the possible inputs of the wire input. 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L126"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AllInputs`
All the encrypted inputs of a given function of a module. 




---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L133"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_inputs_iter`

```python
get_inputs_iter() → Iterable[CompositionClause]
```

Return an iterator over the possible inputs of the wire input. 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L145"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Wire`
A forwarding rule between an output and an input. 




---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L153"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_rules_iter`

```python
get_rules_iter(_) → Iterable[CompositionRule]
```

Return an iterator over composition rules. 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L163"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Wired`
Composition policy which allows the forwarding of certain outputs to certain inputs. 

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L170"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(wires: Optional[set[Wire]] = None)
```








---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L173"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_rules_iter`

```python
get_rules_iter(funcs: list[Graph]) → Iterable[CompositionRule]
```

Return an iterator over composition rules. 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L196"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TracedOutput`
A wrapper type used to trace wiring. 

Allows to tag an output value coming from an other module function, and binds it with information about its origin. 





---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L208"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `WireTracingContextManager`
A context manager returned by the `wire_pipeline` method. 

Activates wire tracing and yields an inputset that can be iterated on for tracing. 

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L215"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(module, inputset)
```









