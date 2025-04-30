<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/extensions/zeros.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.extensions.zeros`
Declaration of `zeros` and `zero` functions, to simplify creation of encrypted zeros. 


---

<a href="../../frontends/concrete-python/concrete/fhe/extensions/zeros.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `zeros`

```python
zeros(shape: Union[int, tuple[int, ]]) → Union[ndarray, Tracer]
```

Create an encrypted array of zeros. 



**Args:**
  shape (Tuple[int, ...]):  shape of the array 



**Returns:**
  Union[np.ndarray, Tracer]:  Tracer that represents the operation during tracing  ndarray filled with zeros otherwise 


---

<a href="../../frontends/concrete-python/concrete/fhe/extensions/zeros.py#L46"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `zero`

```python
zero() → Union[ndarray, Tracer]
```

Create an encrypted scalar with the value of zero. 



**Returns:**
  Union[np.ndarray, Tracer]:  Tracer that respresents the operation during tracing  ndarray with zero otherwise 


---

<a href="../../frontends/concrete-python/concrete/fhe/extensions/zeros.py#L59"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `zeros_like`

```python
zeros_like(array: Union[ndarray, Tracer]) → Union[ndarray, Tracer]
```

Create an encrypted array of zeros with the same shape as another array. 



**Args:**
  array (Union[np.ndarray, Tracer]):  original array 



**Returns:**
  Union[np.ndarray, Tracer]:  Tracer that represent the operation during tracing  ndarray filled with zeros otherwise 


