<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/extensions/round_bit_pattern.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.extensions.round_bit_pattern`
Declaration of `round_bit_pattern` function, to provide an interface for rounded table lookups. 

**Global Variables**
---------------
- **MAXIMUM_TLU_BIT_WIDTH**

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/extensions/round_bit_pattern.py#L157"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `round_bit_pattern`

```python
round_bit_pattern(
    x: Union[int, integer, List, ndarray, Tracer],
    lsbs_to_remove: Union[int, AutoRounder],
    overflow_protection: bool = True
) → Union[int, integer, List, ndarray, Tracer]
```

Round the bit pattern of an integer. 

If `lsbs_to_remove` is an `AutoRounder`:  corresponding integer value will be determined by adjustment process. 

x = 0b_0000_0000 , lsbs_to_remove = 3 => 0b_0000_0000 x = 0b_0000_0001 , lsbs_to_remove = 3 => 0b_0000_0000 x = 0b_0000_0010 , lsbs_to_remove = 3 => 0b_0000_0000 x = 0b_0000_0011 , lsbs_to_remove = 3 => 0b_0000_0000 x = 0b_0000_0100 , lsbs_to_remove = 3 => 0b_0000_1000 x = 0b_0000_0101 , lsbs_to_remove = 3 => 0b_0000_1000 x = 0b_0000_0110 , lsbs_to_remove = 3 => 0b_0000_1000 x = 0b_0000_0111 , lsbs_to_remove = 3 => 0b_0000_1000 

x = 0b_1010_0000 , lsbs_to_remove = 3 => 0b_1010_0000 x = 0b_1010_0001 , lsbs_to_remove = 3 => 0b_1010_0000 x = 0b_1010_0010 , lsbs_to_remove = 3 => 0b_1010_0000 x = 0b_1010_0011 , lsbs_to_remove = 3 => 0b_1010_0000 x = 0b_1010_0100 , lsbs_to_remove = 3 => 0b_1010_1000 x = 0b_1010_0101 , lsbs_to_remove = 3 => 0b_1010_1000 x = 0b_1010_0110 , lsbs_to_remove = 3 => 0b_1010_1000 x = 0b_1010_0111 , lsbs_to_remove = 3 => 0b_1010_1000 

x = 0b_1010_1000 , lsbs_to_remove = 3 => 0b_1010_1000 x = 0b_1010_1001 , lsbs_to_remove = 3 => 0b_1010_1000 x = 0b_1010_1010 , lsbs_to_remove = 3 => 0b_1010_1000 x = 0b_1010_1011 , lsbs_to_remove = 3 => 0b_1010_1000 x = 0b_1010_1100 , lsbs_to_remove = 3 => 0b_1011_0000 x = 0b_1010_1101 , lsbs_to_remove = 3 => 0b_1011_0000 x = 0b_1010_1110 , lsbs_to_remove = 3 => 0b_1011_0000 x = 0b_1010_1111 , lsbs_to_remove = 3 => 0b_1011_0000 

x = 0b_1011_1000 , lsbs_to_remove = 3 => 0b_1011_1000 x = 0b_1011_1001 , lsbs_to_remove = 3 => 0b_1011_1000 x = 0b_1011_1010 , lsbs_to_remove = 3 => 0b_1011_1000 x = 0b_1011_1011 , lsbs_to_remove = 3 => 0b_1011_1000 x = 0b_1011_1100 , lsbs_to_remove = 3 => 0b_1100_0000 x = 0b_1011_1101 , lsbs_to_remove = 3 => 0b_1100_0000 x = 0b_1011_1110 , lsbs_to_remove = 3 => 0b_1100_0000 x = 0b_1011_1111 , lsbs_to_remove = 3 => 0b_1100_0000 



**Args:**
  x (Union[int, np.integer, np.ndarray, Tracer]):  input to round 

 lsbs_to_remove (Union[int, AutoRounder]):  number of the least significant bits to remove  or an auto rounder object which will be used to determine the integer value 

 overflow_protection (bool, default = True)  whether to adjust bit widths and lsbs to remove to avoid overflows 



**Returns:**
  Union[int, np.integer, np.ndarray, Tracer]:  Tracer that respresents the operation during tracing  rounded value(s) otherwise 


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/extensions/round_bit_pattern.py#L24"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Adjusting`
Adjusting class, to be used as early stop signal during adjustment. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/extensions/round_bit_pattern.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(rounder: 'AutoRounder', input_min: int, input_max: int)
```









---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/extensions/round_bit_pattern.py#L40"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AutoRounder`
AutoRounder class, to optimize for number of msbs to keep druing round bit pattern operation. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/extensions/round_bit_pattern.py#L53"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(target_msbs: int = 16)
```








---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/extensions/round_bit_pattern.py#L71"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `adjust`

```python
adjust(
    function: Callable,
    inputset: Union[Iterable[Any], Iterable[Tuple[Any, ]]]
)
```

Adjust AutoRounders in a function using an inputset. 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/extensions/round_bit_pattern.py#L126"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict
```

Dump properties of the rounder to a dict. 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/extensions/round_bit_pattern.py#L140"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load_dict`

```python
load_dict(properties: Dict) → AutoRounder
```

Load previously dumped rounder. 


