<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.compilation.wiring`
Declaration of wiring related class. 

**Global Variables**
---------------
- **TYPE_CHECKING**


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L27"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `NotComposable`
Composition policy that does not allow the forwarding of any output to any input. 




---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L32"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_rules_iter`

```python
get_rules_iter(
    _funcs: List[ForwardRef('FunctionDef')]
) → Iterable[CompositionRule]
```

Return an iterator over composition rules. 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AllComposable`
Composition policy that allows to forward any output of the module to any of its input. 




---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L44"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_rules_iter`

```python
get_rules_iter(funcs: List[Graph]) → Iterable[CompositionRule]
```

Return an iterator over composition rules. 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L63"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `WireOutput`
A protocol for wire outputs. 




---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L68"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_outputs_iter`

```python
get_outputs_iter() → Iterable[CompositionClause]
```

Return an iterator over the possible outputs of the wire output. 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L75"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `WireInput`
A protocol for wire inputs. 




---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L80"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_inputs_iter`

```python
get_inputs_iter() → Iterable[CompositionClause]
```

Return an iterator over the possible inputs of the wire input. 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L86"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Output`
The output of a given function of a module. 




---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L94"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_outputs_iter`

```python
get_outputs_iter() → Iterable[CompositionClause]
```

Return an iterator over the possible outputs of the wire output. 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L101"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AllOutputs`
All the encrypted outputs of a given function of a module. 




---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L108"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_outputs_iter`

```python
get_outputs_iter() → Iterable[CompositionClause]
```

Return an iterator over the possible outputs of the wire output. 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L120"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Input`
The input of a given function of a module. 




---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L128"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_inputs_iter`

```python
get_inputs_iter() → Iterable[CompositionClause]
```

Return an iterator over the possible inputs of the wire input. 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L135"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AllInputs`
All the encrypted inputs of a given function of a module. 




---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L142"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_inputs_iter`

```python
get_inputs_iter() → Iterable[CompositionClause]
```

Return an iterator over the possible inputs of the wire input. 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L154"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Wire`
A forwarding rule between an output and an input. 




---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L162"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_rules_iter`

```python
get_rules_iter(_) → Iterable[CompositionRule]
```

Return an iterator over composition rules. 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L172"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Wired`
Composition policy which allows the forwarding of certain outputs to certain inputs. 

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L179"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(wires: Optional[Set[Wire]] = None)
```








---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L182"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_rules_iter`

```python
get_rules_iter(funcs: List[Graph]) → Iterable[CompositionRule]
```

Return an iterator over composition rules. 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L205"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TracedOutput`
A wrapper type used to trace wiring. 

Allows to tag an output value coming from an other module function, and binds it with information about its origin. 





---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L217"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `WireTracingContextManager`
A context manager returned by the `wire_pipeline` method. 

Activates wire tracing and yields an inputset that can be iterated on for tracing. 

<a href="../../frontends/concrete-python/concrete/fhe/compilation/wiring.py#L224"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(module, inputset)
```









