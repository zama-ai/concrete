<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/wrapper.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.compiler.wrapper`
Wrapper for native Cpp objects. 



---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/wrapper.py#L7"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `WrapperCpp`
Wrapper base class for native Cpp objects. 

Initialization should mainly store the wrapped object, and future calls to the wrapper will be forwarded to it. A static wrap method is provided to be more explicit. Wrappers should always be constructed using the new method, which construct the Cpp object using the provided arguments, then wrap it. Classes that inherit from this class should preferably type check the wrapped object during calls to init, and reimplement the new method if the class is meant to be constructed. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/wrapper.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(cpp_obj)
```








---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/wrapper.py#L37"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `cpp`

```python
cpp()
```

Return the Cpp wrapped object. 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/wrapper.py#L32"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `new`

```python
new(*args, **kwargs)
```

Create a new wrapper by building the underlying object with a specific set of arguments. 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/wrapper.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `wrap`

```python
wrap(cpp_obj) → WrapperCpp
```

Wrap the Cpp object into a Python object. 



**Args:**
 
 - <b>`cpp_obj`</b>:  object to wrap 



**Returns:**
 
 - <b>`WrapperCpp`</b>:  wrapper 


