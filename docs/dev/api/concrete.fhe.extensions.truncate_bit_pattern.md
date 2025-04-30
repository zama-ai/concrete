<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/extensions/truncate_bit_pattern.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.extensions.truncate_bit_pattern`
Declaration of `truncate_bit_pattern` extension. 

**Global Variables**
---------------
- **MAXIMUM_TLU_BIT_WIDTH**

---

<a href="../../frontends/concrete-python/concrete/fhe/extensions/truncate_bit_pattern.py#L173"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `truncate_bit_pattern`

```python
truncate_bit_pattern(
    x: Union[int, integer, list, ndarray, Tracer],
    lsbs_to_remove: Union[int, AutoTruncator]
) → Union[int, integer, list, ndarray, Tracer]
```

Round the bit pattern of an integer. 

If `lsbs_to_remove` is an `AutoTruncator`:  corresponding integer value will be determined by adjustment process. 

x = 0b_0000 , lsbs_to_remove = 2 => 0b_0000 x = 0b_0001 , lsbs_to_remove = 2 => 0b_0000 x = 0b_0010 , lsbs_to_remove = 2 => 0b_0000 x = 0b_0100 , lsbs_to_remove = 2 => 0b_0100 x = 0b_0110 , lsbs_to_remove = 2 => 0b_0100 x = 0b_1100 , lsbs_to_remove = 2 => 0b_1100 x = 0b_abcd , lsbs_to_remove = 2 => 0b_ab00 



**Args:**
  x (Union[int, np.integer, np.ndarray, Tracer]):  input to truncate 

 lsbs_to_remove (Union[int, AutoTruncator]):  number of the least significant bits to clear  or an auto truncator object which will be used to determine the integer value 



**Returns:**
  Union[int, np.integer, np.ndarray, Tracer]:  Tracer that represents the operation during tracing  truncated value(s) otherwise 


---

<a href="../../frontends/concrete-python/concrete/fhe/extensions/truncate_bit_pattern.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Adjusting`
Adjusting class, to be used as early stop signal during adjustment. 

<a href="../../frontends/concrete-python/concrete/fhe/extensions/truncate_bit_pattern.py#L34"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(truncator: 'AutoTruncator', input_min: int, input_max: int)
```









---

<a href="../../frontends/concrete-python/concrete/fhe/extensions/truncate_bit_pattern.py#L41"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AutoTruncator`
AutoTruncator class, to optimize for the number of msbs to keep during truncate operation. 

<a href="../../frontends/concrete-python/concrete/fhe/extensions/truncate_bit_pattern.py#L54"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(target_msbs: int = 16)
```








---

<a href="../../frontends/concrete-python/concrete/fhe/extensions/truncate_bit_pattern.py#L72"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `adjust`

```python
adjust(
    function: Callable,
    inputset: Union[Iterable[Any], Iterable[tuple[Any, ]]]
)
```

Adjust AutoTruncators in a function using an inputset. 

---

<a href="../../frontends/concrete-python/concrete/fhe/extensions/truncate_bit_pattern.py#L142"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → dict
```

Dump properties of the truncator to a dict. 

---

<a href="../../frontends/concrete-python/concrete/fhe/extensions/truncate_bit_pattern.py#L156"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load_dict`

```python
load_dict(properties: dict) → AutoTruncator
```

Load previously dumped truncator. 


