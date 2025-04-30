<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/extensions/convolution.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.extensions.convolution`
Tracing and evaluation of convolution. 

**Global Variables**
---------------
- **SUPPORTED_AUTO_PAD**

---

<a href="../../frontends/concrete-python/concrete/fhe/extensions/convolution.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `conv`

```python
conv(
    x: Union[ndarray, Tracer],
    weight: Union[ndarray, list, Tracer],
    bias: Optional[ndarray, list, Tracer] = None,
    pads: Optional[tuple[int, ], list[int]] = None,
    strides: Optional[tuple[int, ], list[int]] = None,
    dilations: Optional[tuple[int, ], list[int]] = None,
    kernel_shape: Optional[tuple[int, ], list[int]] = None,
    group: int = 1,
    auto_pad: str = 'NOTSET'
) → Union[ndarray, Tracer]
```

Trace and evaluate convolution operations. 

Refer to https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv for more info. 



**Args:**
 
 - <b>`x`</b> (Union[np.ndarray, Tracer]):  input of shape (N, C, D1, ..., DN) 
 - <b>`weight`</b> (Union[np.ndarray, Tracer]):  kernel of shape (F, C / group, K1, ..., KN) 
 - <b>`bias`</b> (Optional[Union[np.ndarray, Tracer]], optional):  bias of shape (F,). Defaults to None. pads (Optional[Union[Tuple[int, ...], List[int]]], optional):  padding for the beginning and ending along each spatial axis  (D1_begin, D2_begin, ..., D1_end, D2_end, ...).  Will be set to 0 along each spatial axis if not set. strides (Optional[Union[Tuple[int, ...], List[int]]], optional):  stride along each spatial axis. Will be set to 1 along each spatial axis if not set. dilations (Optional[Union[Tuple[int, ...], List[int]]], optional):  dilation along each spatial axis. Will be set to 1 along each spatial axis if not set. kernel_shape (Optional[Union[Tuple[int, ...], List[int]]], optional):  shape of the convolution kernel. Inferred from input weight if not present group (int, optional):  number of groups input channels and output channels are divided into. Defaults to 1. 
 - <b>`auto_pad`</b> (str, optional):  padding strategy. Defaults to "NOTSET". 



**Raises:**
 
 - <b>`ValueError`</b>:  if arguments are not appropriate 
 - <b>`TypeError`</b>:  unexpected types 
 - <b>`NotImplementedError`</b>:  a convolution that we don't support 



**Returns:**
 
 - <b>`Union[np.ndarray, Tracer]`</b>:  evaluation result or traced computation 


