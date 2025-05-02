<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/extensions/ones.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.extensions.ones`
Declaration of `ones` and `one` functions, to simplify creation of encrypted ones. 


---

<a href="../../frontends/concrete-python/concrete/fhe/extensions/ones.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `ones`

```python
ones(shape: Union[int, tuple[int, ]]) → Union[ndarray, Tracer]
```

Create an encrypted array of ones. 



**Args:**
  shape (Tuple[int, ...]):  shape of the array 



**Returns:**
  Union[np.ndarray, Tracer]:  Tracer that represents the operation during tracing  ndarray filled with ones otherwise 


---

<a href="../../frontends/concrete-python/concrete/fhe/extensions/ones.py#L46"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `one`

```python
one() → Union[ndarray, Tracer]
```

Create an encrypted scalar with the value of one. 



**Returns:**
  Union[np.ndarray, Tracer]:  Tracer that represents the operation during tracing  ndarray with one otherwise 


---

<a href="../../frontends/concrete-python/concrete/fhe/extensions/ones.py#L59"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `ones_like`

```python
ones_like(array: Union[ndarray, Tracer]) → Union[ndarray, Tracer]
```

Create an encrypted array of ones with the same shape as another array. 



**Args:**
  array (Union[np.ndarray, Tracer]):  original array 



**Returns:**
  Union[np.ndarray, Tracer]:  Tracer that represent the operation during tracing  ndarray filled with ones otherwise 


