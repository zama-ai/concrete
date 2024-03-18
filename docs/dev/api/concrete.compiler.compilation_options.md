<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_options.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.compiler.compilation_options`
CompilationOptions. 



---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_options.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CompilationOptions`
CompilationOptions holds different flags and options of the compilation process. 

It controls different parallelization flags, diagnostic verification, and also the name of entrypoint function. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_options.py#L29"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(compilation_options: CompilationOptions)
```

Wrap the native Cpp object. 



**Args:**
 
 - <b>`compilation_options`</b> (_CompilationOptions):  object to wrap 



**Raises:**
 
 - <b>`TypeError`</b>:  if compilation_options is not of type _CompilationOptions 




---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_options.py#L400"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `force_encoding`

```python
force_encoding(encoding: Encoding)
```

Force the compiler to use a specific encoding. 



**Args:**
 
 - <b>`encoding`</b> (Encoding):  the encoding to force the compiler to use 



**Raises:**
 
 - <b>`TypeError`</b>:  if encoding is not of type Encoding 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_options.py#L44"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `new`

```python
new(function_name='main', backend=<Backend.CPU: 0>) → CompilationOptions
```

Build a CompilationOptions. 



**Args:**
 
 - <b>`function_name`</b> (str, optional):  name of the entrypoint function. Defaults to "main". 



**Raises:**
 
 - <b>`TypeError`</b>:  if function_name is not an str 



**Returns:**
 CompilationOptions 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_options.py#L315"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `set_all_v0_parameter`

```python
set_all_v0_parameter(
    glwe_dim: int,
    log_poly_size: int,
    n_small: int,
    br_level: int,
    br_log_base: int,
    ks_level: int,
    ks_log_base: int,
    crt_decomp: List[int],
    cbs_level: int,
    cbs_log_base: int,
    pks_level: int,
    pks_log_base: int,
    pks_input_lwe_dim: int,
    pks_output_poly_size: int
)
```

Set all the V0 parameters. 



**Args:**
 
 - <b>`glwe_dim`</b> (int):  GLWE dimension 
 - <b>`log_poly_size`</b> (int):  log of polynomial size 
 - <b>`n_small`</b> (int):  n 
 - <b>`br_level`</b> (int):  bootstrap level 
 - <b>`br_log_base`</b> (int):  bootstrap base log 
 - <b>`ks_level`</b> (int):  keyswitch level 
 - <b>`ks_log_base`</b> (int):  keyswitch base log 
 - <b>`crt_decomp`</b> (List[int]):  CRT decomposition vector 
 - <b>`cbs_level`</b> (int):  circuit bootstrap level 
 - <b>`cbs_log_base`</b> (int):  circuit bootstrap base log 
 - <b>`pks_level`</b> (int):  packing keyswitch level 
 - <b>`pks_log_base`</b> (int):  packing keyswitch base log 
 - <b>`pks_input_lwe_dim`</b> (int):  packing keyswitch input LWE dimension 
 - <b>`pks_output_poly_size`</b> (int):  packing keyswitch output polynomial size 



**Raises:**
 
 - <b>`TypeError`</b>:  if parameters are not of type int 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_options.py#L83"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `set_auto_parallelize`

```python
set_auto_parallelize(auto_parallelize: bool)
```

Set option for auto parallelization. 



**Args:**
 
 - <b>`auto_parallelize`</b> (bool):  whether to turn it on or off 



**Raises:**
 
 - <b>`TypeError`</b>:  if the value to set is not boolean 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_options.py#L439"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `set_batch_tfhe_ops`

```python
set_batch_tfhe_ops(batch_tfhe_ops: bool)
```

Set flag that triggers the batching of scalar TFHE operations. 



**Args:**
 
 - <b>`batch_tfhe_ops`</b> (bool):  whether to batch tfhe ops. 



**Raises:**
 
 - <b>`TypeError`</b>:  if the value to set is not bool 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_options.py#L70"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `set_composable`

```python
set_composable(composable: bool)
```

Set option for composition. 



**Args:**
 
 - <b>`composable`</b> (bool):  whether to turn it on or off 



**Raises:**
 
 - <b>`TypeError`</b>:  if the value to set is not boolean 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_options.py#L109"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `set_compress_evaluation_keys`

```python
set_compress_evaluation_keys(compress_evaluation_keys: bool)
```

Set option for compression of evaluation keys. 



**Args:**
 
 - <b>`compress_evaluation_keys`</b> (bool):  whether to turn it on or off 



**Raises:**
 
 - <b>`TypeError`</b>:  if the value to set is not boolean 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_options.py#L135"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `set_dataflow_parallelize`

```python
set_dataflow_parallelize(dataflow_parallelize: bool)
```

Set option for dataflow parallelization. 



**Args:**
 
 - <b>`dataflow_parallelize`</b> (bool):  whether to turn it on or off 



**Raises:**
 
 - <b>`TypeError`</b>:  if the value to set is not boolean 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_options.py#L192"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `set_display_optimizer_choice`

```python
set_display_optimizer_choice(display: bool)
```

Set display flag of optimizer choices. 



**Args:**
 
 - <b>`display`</b> (bool):  if true the compiler display optimizer choices 



**Raises:**
 
 - <b>`TypeError`</b>:  if the value is not a bool 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_options.py#L426"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `set_emit_gpu_ops`

```python
set_emit_gpu_ops(emit_gpu_ops: bool)
```

Set flag that allows gpu ops to be emitted. 



**Args:**
 
 - <b>`emit_gpu_ops`</b> (bool):  whether to emit gpu ops. 



**Raises:**
 
 - <b>`TypeError`</b>:  if the value to set is not bool 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_options.py#L161"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `set_funcname`

```python
set_funcname(funcname: str)
```

Set entrypoint function name. 



**Args:**
 
 - <b>`funcname`</b> (str):  name of the entrypoint function 



**Raises:**
 
 - <b>`TypeError`</b>:  if the value to set is not str 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_options.py#L233"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `set_global_p_error`

```python
set_global_p_error(global_p_error: float)
```

Set global error probability for the full circuit. 



**Args:**
 
 - <b>`global_p_error`</b> (float):  probability of error for the full circuit 



**Raises:**
 
 - <b>`TypeError`</b>:  if the value to set is not float 
 - <b>`ValueError`</b>:  if the value to set is not in interval ]0; 1] 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_options.py#L96"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `set_loop_parallelize`

```python
set_loop_parallelize(loop_parallelize: bool)
```

Set option for loop parallelization. 



**Args:**
 
 - <b>`loop_parallelize`</b> (bool):  whether to turn it on or off 



**Raises:**
 
 - <b>`TypeError`</b>:  if the value to set is not boolean 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_options.py#L148"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `set_optimize_concrete`

```python
set_optimize_concrete(optimize: bool)
```

Set flag to enable/disable optimization of concrete intermediate representation. 



**Args:**
 
 - <b>`optimize`</b> (bool):  whether to turn it on or off 



**Raises:**
 
 - <b>`TypeError`</b>:  if the value to set is not boolean 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_options.py#L218"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `set_optimizer_multi_parameter_strategy`

```python
set_optimizer_multi_parameter_strategy(
    strategy: OptimizerMultiParameterStrategy
)
```

Set the strategy of the optimizer for multi-parameter. 



**Args:**
 
 - <b>`strategy`</b> (OptimizerMultiParameterStrategy):  Use the specified optmizer multi-parameter strategy. 



**Raises:**
 
 - <b>`TypeError`</b>:  if the value is not a OptimizerMultiParameterStrategy 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_options.py#L205"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `set_optimizer_strategy`

```python
set_optimizer_strategy(strategy: OptimizerStrategy)
```

Set the strategy of the optimizer. 



**Args:**
 
 - <b>`strategy`</b> (OptimizerStrategy):  Use the specified optmizer strategy. 



**Raises:**
 
 - <b>`TypeError`</b>:  if the value is not an OptimizerStrategy 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_options.py#L174"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `set_p_error`

```python
set_p_error(p_error: float)
```

Set error probability for shared by each pbs. 



**Args:**
 
 - <b>`p_error`</b> (float):  probability of error for each lut 



**Raises:**
 
 - <b>`TypeError`</b>:  if the value to set is not float 
 - <b>`ValueError`</b>:  if the value to set is not in interval ]0; 1] 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_options.py#L251"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `set_security_level`

```python
set_security_level(security_level: int)
```

Set security level. 



**Args:**
 
 - <b>`security_level`</b> (int):  the target number of bits of security to compile the circuit 



**Raises:**
 
 - <b>`TypeError`</b>:  if the value to set is not int 
 - <b>`ValueError`</b>:  if the value to set is not in interval ]0; 1] 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_options.py#L265"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `set_v0_parameter`

```python
set_v0_parameter(
    glwe_dim: int,
    log_poly_size: int,
    n_small: int,
    br_level: int,
    br_log_base: int,
    ks_level: int,
    ks_log_base: int
)
```

Set the basic V0 parameters. 



**Args:**
 
 - <b>`glwe_dim`</b> (int):  GLWE dimension 
 - <b>`log_poly_size`</b> (int):  log of polynomial size 
 - <b>`n_small`</b> (int):  n 
 - <b>`br_level`</b> (int):  bootstrap level 
 - <b>`br_log_base`</b> (int):  bootstrap base log 
 - <b>`ks_level`</b> (int):  keyswitch level 
 - <b>`ks_log_base`</b> (int):  keyswitch base log 



**Raises:**
 
 - <b>`TypeError`</b>:  if parameters are not of type int 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_options.py#L122"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `set_verify_diagnostics`

```python
set_verify_diagnostics(verify_diagnostics: bool)
```

Set option for diagnostics verification. 



**Args:**
 
 - <b>`verify_diagnostics`</b> (bool):  whether to turn it on or off 



**Raises:**
 
 - <b>`TypeError`</b>:  if the value to set is not boolean 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/compilation_options.py#L413"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `simulation`

```python
simulation(simulate: bool)
```

Enable or disable simulation. 



**Args:**
 
 - <b>`simulate`</b> (bool):  flag to enable or disable simulation 



**Raises:**
 
 - <b>`TypeError`</b>:  if the value to set is not bool 


