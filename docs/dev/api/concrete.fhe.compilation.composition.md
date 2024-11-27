<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/compilation/composition.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.compilation.composition`
Declaration of classes related to composition. 



---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/composition.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CompositionClause`
A raw composition clause. 




---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/composition.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `create`

```python
create(tup: Tuple[str, int]) → CompositionClause
```

Create a composition clause from a tuple of a function name and a position. 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/composition.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CompositionRule`
A raw composition rule. 




---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/composition.py#L36"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `create`

```python
create(tup: Tuple[CompositionClause, CompositionClause]) → CompositionRule
```

Create a composition rule from a tuple containing an output clause and an input clause. 


---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/composition.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CompositionPolicy`
A protocol for composition policies. 




---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/composition.py#L50"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_rules_iter`

```python
get_rules_iter(funcs: List[Graph]) → Iterable[CompositionRule]
```

Return an iterator over composition rules. 


