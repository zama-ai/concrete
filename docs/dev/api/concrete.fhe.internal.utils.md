<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/internal/utils.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.internal.utils`
Declaration of various functions and constants related to the entire project. 


---

<a href="../../frontends/concrete-python/concrete/fhe/internal/utils.py#L6"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `assert_that`

```python
assert_that(condition: bool, message: str = '')
```

Assert a condition. 



**Args:**
  condition (bool):  condition to assert 

 message (str):  message to give to `AssertionError` if the condition does not hold 



**Raises:**
  AssertionError:  if the condition does not hold 


---

<a href="../../frontends/concrete-python/concrete/fhe/internal/utils.py#L26"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `unreachable`

```python
unreachable()
```

Raise a RuntimeError to indicate unreachable code is entered. 


