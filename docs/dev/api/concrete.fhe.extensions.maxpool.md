<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/extensions/maxpool.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.extensions.maxpool`
Tracing and evaluation of maxpool. 

**Global Variables**
---------------
- **AVAILABLE_AUTO_PAD**
- **AVAILABLE_CEIL_MODE**
- **AVAILABLE_STORAGE_ORDER**
- **SUPPORTED_AUTO_PAD**
- **SUPPORTED_CEIL_MODE**
- **SUPPORTED_STORAGE_ORDER**

---

<a href="../../frontends/concrete-python/concrete/fhe/extensions/maxpool.py#L61"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `maxpool`

```python
maxpool(
    x: Union[ndarray, Tracer],
    kernel_shape: Union[tuple[int, ], list[int]],
    strides: Optional[tuple[int, ], list[int]] = None,
    auto_pad: str = 'NOTSET',
    pads: Optional[tuple[int, ], list[int]] = None,
    dilations: Optional[tuple[int, ], list[int]] = None,
    ceil_mode: int = 0,
    storage_order: int = 0
) → Union[ndarray, Tracer]
```

Evaluate or trace MaxPool operation. 

Refer to https://github.com/onnx/onnx/blob/main/docs/Operators.md#maxpool for more info. 



**Args:**
  x (Union[np.ndarray, Tracer]):  input of shape (N, C, D1, ..., DN) 

 kernel_shape (Union[Tuple[int, ...], List[int]]):  shape of the kernel 

 strides (Optional[Union[Tuple[int, ...], List[int]]]):  stride along each spatial axis  set to 1 along each spatial axis if not set 

 auto_pad (str, default = "NOTSET"):  padding strategy 

 pads (Optional[Union[Tuple[int, ...], List[int]]]):  padding for the beginning and ending along each spatial axis  (D1_begin, D2_begin, ..., D1_end, D2_end, ...)  set to 0 along each spatial axis if not set 

 dilations (Optional[Union[Tuple[int, ...], List[int]]]):  dilation along each spatial axis  set to 1 along each spatial axis if not set 

 ceil_mode (int, default = 1):  ceiling mode 

 storage_order (int, default = 0):  storage order, 0 for row major, 1 for column major 



**Raises:**
  TypeError:  if arguments are inappropriately typed 

 ValueError:  if arguments are inappropriate 

 NotImplementedError:  if desired operation is not supported yet 



**Returns:**
  Union[np.ndarray, Tracer]:  maxpool over the input or traced computation 


