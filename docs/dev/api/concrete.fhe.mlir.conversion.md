<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/mlir/conversion.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.mlir.conversion`
Declaration of `ConversionType` and `Conversion` classes. 



---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/conversion.py#L27"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ConversionType`
ConversionType class, to make it easier to work with MLIR types. 

<a href="../../frontends/concrete-python/concrete/fhe/mlir/conversion.py#L41"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(mlir: Type)
```






---

#### <kbd>property</kbd> is_clear





---

#### <kbd>property</kbd> is_scalar





---

#### <kbd>property</kbd> is_tensor





---

#### <kbd>property</kbd> is_unsigned








---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/conversion.py#L139"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Conversion`
Conversion class, to store MLIR operations with additional information. 

<a href="../../frontends/concrete-python/concrete/fhe/mlir/conversion.py#L151"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(origin: Node, result: OpResult)
```






---

#### <kbd>property</kbd> bit_width





---

#### <kbd>property</kbd> is_clear





---

#### <kbd>property</kbd> is_encrypted





---

#### <kbd>property</kbd> is_scalar





---

#### <kbd>property</kbd> is_signed





---

#### <kbd>property</kbd> is_tensor





---

#### <kbd>property</kbd> is_unsigned





---

#### <kbd>property</kbd> original_bit_width

Get the original bit-width of the conversion. 

If not explicitly set, defaults to the actual bit width. 

---

#### <kbd>property</kbd> shape







---

<a href="../../frontends/concrete-python/concrete/fhe/mlir/conversion.py#L159"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `set_original_bit_width`

```python
set_original_bit_width(original_bit_width: int)
```

Set the original bit-width of the conversion. 


