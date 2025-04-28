<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/__init__.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.compiler`
Compiler submodule. 

**Global Variables**
---------------
- **utils**: #  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete/blob/main/LICENSE.txt for license information.

- **compilation_feedback**: #  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete/blob/main/LICENSE.txt for license information.

- **compilation_context**: #  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete/blob/main/LICENSE.txt for license information.

- **tfhers_int**

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/__init__.py#L62"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `init_dfr`

```python
init_dfr()
```

Initialize dataflow parallelization. 

It is not always required to initialize the dataflow runtime as it can be implicitely done during compilation. However, it is required in case no compilation has previously been done and the runtime is needed 


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/__init__.py#L71"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `check_gpu_enabled`

```python
check_gpu_enabled() → bool
```

Check whether the compiler and runtime support GPU offloading. 

GPU offloading is not always available, in particular in non-GPU wheels. 


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/__init__.py#L78"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `check_gpu_available`

```python
check_gpu_available() → bool
```

Check whether a CUDA device is available and online. 


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/__init__.py#L88"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `round_trip`

```python
round_trip(mlir_str: str) → str
```

Parse the MLIR input, then return it back. 

Useful to check the validity of an MLIR representation 



**Args:**
 
 - <b>`mlir_str`</b> (str):  textual representation of an MLIR code 



**Raises:**
 
 - <b>`TypeError`</b>:  if mlir_str is not of type str 



**Returns:**
 
 - <b>`str`</b>:  textual representation of the MLIR code after parsing 


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/__init__.py#L107"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `RangeRestrictionHandler`
Handler to serialize and deserialize range restrictions 




---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/__init__.py#L111"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `flatten`

```python
flatten(obj, data)
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/__init__.py#L115"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `restore`

```python
restore(obj)
```






---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/__init__.py#L119"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `KeysetRestrictionHandler`
Handler to serialize and deserialize keyset restrictions 




---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/__init__.py#L123"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `flatten`

```python
flatten(obj, data)
```





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/__init__.py#L127"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `restore`

```python
restore(obj)
```






