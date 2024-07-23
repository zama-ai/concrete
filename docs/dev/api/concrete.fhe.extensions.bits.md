<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/extensions/bits.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.extensions.bits`
Bit extraction extensions. 

**Global Variables**
---------------
- **MIN_EXTRACTABLE_BIT**
- **MAX_EXTRACTABLE_BIT**

---

<a href="../../frontends/concrete-python/concrete/fhe/extensions/bits.py#L155"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `bits`

```python
bits(x: Union[int, integer, list, ndarray, Tracer]) → Bits
```

Extract bits of integers. 



**Args:**
  x (Union[int, np.integer, list, np.ndarray, Tracer]):  input to extract bits from 



**Returns:**
  Bits:  Auxiliary class to represent bits of the integer 


---

<a href="../../frontends/concrete-python/concrete/fhe/extensions/bits.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Bits`
Bits class, to provide indexing into the bits of integers. 

<a href="../../frontends/concrete-python/concrete/fhe/extensions/bits.py#L26"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(value: Union[int, integer, list, ndarray, Tracer])
```









