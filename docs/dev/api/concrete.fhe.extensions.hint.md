<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/extensions/hint.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.extensions.hint`
Declaration of hinting extensions, to provide more information to Concrete. 


---

<a href="../../frontends/concrete-python/concrete/fhe/extensions/hint.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `hint`

```python
hint(
    x: Union[Tracer, Any],
    bit_width: Optional[int] = None,
    can_store: Optional[Any] = None
) → Union[Tracer, Any]
```

Hint the compilation process about properties of a value. 

Hints are useful if you know something about a value, but it's hard to cover in the inputset. An example of this can be a complex circuit doing a lot of bitwise operations on 8-bits. It's very hard to make sure every intermediate has 8-bits, but you can use hints to solve this. If you mark your intermediates using this function to be 8-bits, they'll be assigned at least 8-bits during the bit-width assignment step. 



**Args:**
  x (Union[Tracer, Any]):  value to hint 

 bit_width (Optional[int], default = None):  hint about bit width 

 can_store (Optional[Any], default = None):  hint that the value needs to be able to store the given value 



**Returns:**
  Union[Tracer, Any]:  hinted value 


