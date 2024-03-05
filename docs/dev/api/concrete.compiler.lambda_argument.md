<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/lambda_argument.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.compiler.lambda_argument`
LambdaArgument. 

**Global Variables**
---------------
- **ACCEPTED_INTS**


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/lambda_argument.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LambdaArgument`
LambdaArgument holds scalar or tensor values. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/lambda_argument.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(lambda_argument: LambdaArgument)
```

Wrap the native Cpp object. 



**Args:**
 
 - <b>`lambda_argument`</b> (_LambdaArgument):  object to wrap 



**Raises:**
 
 - <b>`TypeError`</b>:  if lambda_argument is not of type _LambdaArgument 




---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/lambda_argument.py#L46"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `from_scalar`

```python
from_scalar(scalar: int) → LambdaArgument
```

Build a LambdaArgument containing the given scalar value. 



**Args:**
 
 - <b>`scalar`</b> (int or numpy.uint):  scalar value to embed in LambdaArgument 



**Raises:**
 
 - <b>`TypeError`</b>:  if scalar is not of type int or numpy.uint 



**Returns:**
 LambdaArgument 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/lambda_argument.py#L65"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `from_signed_scalar`

```python
from_signed_scalar(scalar: int) → LambdaArgument
```

Build a LambdaArgument containing the given scalar value. 



**Args:**
 
 - <b>`scalar`</b> (int or numpy.int):  scalar value to embed in LambdaArgument 



**Raises:**
 
 - <b>`TypeError`</b>:  if scalar is not of type int or numpy.uint 



**Returns:**
 LambdaArgument 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/lambda_argument.py#L149"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `from_tensor_i16`

```python
from_tensor_i16(data: List[int], shape: List[int]) → LambdaArgument
```

Build a LambdaArgument containing the given tensor. 



**Args:**
 
 - <b>`data`</b> (List[int]):  flattened tensor data 
 - <b>`shape`</b> (List[int]):  shape of original tensor before flattening 



**Returns:**
 LambdaArgument 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/lambda_argument.py#L162"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `from_tensor_i32`

```python
from_tensor_i32(data: List[int], shape: List[int]) → LambdaArgument
```

Build a LambdaArgument containing the given tensor. 



**Args:**
 
 - <b>`data`</b> (List[int]):  flattened tensor data 
 - <b>`shape`</b> (List[int]):  shape of original tensor before flattening 



**Returns:**
 LambdaArgument 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/lambda_argument.py#L175"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `from_tensor_i64`

```python
from_tensor_i64(data: List[int], shape: List[int]) → LambdaArgument
```

Build a LambdaArgument containing the given tensor. 



**Args:**
 
 - <b>`data`</b> (List[int]):  flattened tensor data 
 - <b>`shape`</b> (List[int]):  shape of original tensor before flattening 



**Returns:**
 LambdaArgument 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/lambda_argument.py#L136"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `from_tensor_i8`

```python
from_tensor_i8(data: List[int], shape: List[int]) → LambdaArgument
```

Build a LambdaArgument containing the given tensor. 



**Args:**
 
 - <b>`data`</b> (List[int]):  flattened tensor data 
 - <b>`shape`</b> (List[int]):  shape of original tensor before flattening 



**Returns:**
 LambdaArgument 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/lambda_argument.py#L97"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `from_tensor_u16`

```python
from_tensor_u16(data: List[int], shape: List[int]) → LambdaArgument
```

Build a LambdaArgument containing the given tensor. 



**Args:**
 
 - <b>`data`</b> (List[int]):  flattened tensor data 
 - <b>`shape`</b> (List[int]):  shape of original tensor before flattening 



**Returns:**
 LambdaArgument 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/lambda_argument.py#L110"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `from_tensor_u32`

```python
from_tensor_u32(data: List[int], shape: List[int]) → LambdaArgument
```

Build a LambdaArgument containing the given tensor. 



**Args:**
 
 - <b>`data`</b> (List[int]):  flattened tensor data 
 - <b>`shape`</b> (List[int]):  shape of original tensor before flattening 



**Returns:**
 LambdaArgument 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/lambda_argument.py#L123"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `from_tensor_u64`

```python
from_tensor_u64(data: List[int], shape: List[int]) → LambdaArgument
```

Build a LambdaArgument containing the given tensor. 



**Args:**
 
 - <b>`data`</b> (List[int]):  flattened tensor data 
 - <b>`shape`</b> (List[int]):  shape of original tensor before flattening 



**Returns:**
 LambdaArgument 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/lambda_argument.py#L84"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `from_tensor_u8`

```python
from_tensor_u8(data: List[int], shape: List[int]) → LambdaArgument
```

Build a LambdaArgument containing the given tensor. 



**Args:**
 
 - <b>`data`</b> (List[int]):  flattened tensor data 
 - <b>`shape`</b> (List[int]):  shape of original tensor before flattening 



**Returns:**
 LambdaArgument 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/lambda_argument.py#L204"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_scalar`

```python
get_scalar() → int
```

Return the contained scalar value. 



**Returns:**
  int 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/lambda_argument.py#L212"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_signed_scalar`

```python
get_signed_scalar() → int
```

Return the contained scalar value. 



**Returns:**
  int 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/lambda_argument.py#L244"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_signed_tensor_data`

```python
get_signed_tensor_data() → List[int]
```

Return the contained flattened tensor data. 



**Returns:**
  List[int] 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/lambda_argument.py#L236"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_tensor_data`

```python
get_tensor_data() → List[int]
```

Return the contained flattened tensor data. 



**Returns:**
  List[int] 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/lambda_argument.py#L228"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_tensor_shape`

```python
get_tensor_shape() → List[int]
```

Return the shape of the contained tensor. 



**Returns:**
 
 - <b>`List[int]`</b>:  tensor shape 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/lambda_argument.py#L196"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_scalar`

```python
is_scalar() → bool
```

Check if the contained argument is a scalar. 



**Returns:**
  bool 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/lambda_argument.py#L188"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_signed`

```python
is_signed() → bool
```

Check if the contained argument is signed. 



**Returns:**
  bool 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/lambda_argument.py#L220"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `is_tensor`

```python
is_tensor() → bool
```

Check if the contained argument is a tensor. 



**Returns:**
  bool 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/lambda_argument.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `new`

```python
new(*args, **kwargs)
```

Use from_scalar or from_tensor instead. 



**Raises:**
  RuntimeError 


