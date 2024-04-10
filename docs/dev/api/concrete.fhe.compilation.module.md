<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/module.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.compilation.module`
Declaration of `FheModule` classes. 



---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/module.py#L31"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ExecutionRt`
Runtime object class for execution. 





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/module.py#L40"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SimulationRt`
Runtime object class for simulation. 





---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/module.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FheFunction`
Fhe function class, allowing to run or simulate one function of an fhe module. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/module.py#L57"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(name: str, runtime: Union[ExecutionRt, SimulationRt], graph: Graph)
```






---

#### <kbd>property</kbd> clear_addition_count

Get the number of clear additions in the function. 

---

#### <kbd>property</kbd> clear_addition_count_per_parameter

Get the number of clear additions per parameter in the function. 

---

#### <kbd>property</kbd> clear_addition_count_per_tag

Get the number of clear additions per tag in the function. 

---

#### <kbd>property</kbd> clear_addition_count_per_tag_per_parameter

Get the number of clear additions per tag per parameter in the function. 

---

#### <kbd>property</kbd> clear_multiplication_count

Get the number of clear multiplications in the function. 

---

#### <kbd>property</kbd> clear_multiplication_count_per_parameter

Get the number of clear multiplications per parameter in the function. 

---

#### <kbd>property</kbd> clear_multiplication_count_per_tag

Get the number of clear multiplications per tag in the function. 

---

#### <kbd>property</kbd> clear_multiplication_count_per_tag_per_parameter

Get the number of clear multiplications per tag per parameter in the function. 

---

#### <kbd>property</kbd> encrypted_addition_count

Get the number of encrypted additions in the function. 

---

#### <kbd>property</kbd> encrypted_addition_count_per_parameter

Get the number of encrypted additions per parameter in the function. 

---

#### <kbd>property</kbd> encrypted_addition_count_per_tag

Get the number of encrypted additions per tag in the function. 

---

#### <kbd>property</kbd> encrypted_addition_count_per_tag_per_parameter

Get the number of encrypted additions per tag per parameter in the function. 

---

#### <kbd>property</kbd> encrypted_negation_count

Get the number of encrypted negations in the function. 

---

#### <kbd>property</kbd> encrypted_negation_count_per_parameter

Get the number of encrypted negations per parameter in the function. 

---

#### <kbd>property</kbd> encrypted_negation_count_per_tag

Get the number of encrypted negations per tag in the function. 

---

#### <kbd>property</kbd> encrypted_negation_count_per_tag_per_parameter

Get the number of encrypted negations per tag per parameter in the function. 

---

#### <kbd>property</kbd> key_switch_count

Get the number of key switches in the function. 

---

#### <kbd>property</kbd> key_switch_count_per_parameter

Get the number of key switches per parameter in the function. 

---

#### <kbd>property</kbd> key_switch_count_per_tag

Get the number of key switches per tag in the function. 

---

#### <kbd>property</kbd> key_switch_count_per_tag_per_parameter

Get the number of key switches per tag per parameter in the function. 

---

#### <kbd>property</kbd> packing_key_switch_count

Get the number of packing key switches in the function. 

---

#### <kbd>property</kbd> packing_key_switch_count_per_parameter

Get the number of packing key switches per parameter in the function. 

---

#### <kbd>property</kbd> packing_key_switch_count_per_tag

Get the number of packing key switches per tag in the function. 

---

#### <kbd>property</kbd> packing_key_switch_count_per_tag_per_parameter

Get the number of packing key switches per tag per parameter in the function. 

---

#### <kbd>property</kbd> programmable_bootstrap_count

Get the number of programmable bootstraps in the function. 

---

#### <kbd>property</kbd> programmable_bootstrap_count_per_parameter

Get the number of programmable bootstraps per bit width in the function. 

---

#### <kbd>property</kbd> programmable_bootstrap_count_per_tag

Get the number of programmable bootstraps per tag in the function. 

---

#### <kbd>property</kbd> programmable_bootstrap_count_per_tag_per_parameter

Get the number of programmable bootstraps per tag per bit width in the function. 

---

#### <kbd>property</kbd> size_of_inputs

Get size of the inputs of the function. 

---

#### <kbd>property</kbd> size_of_outputs

Get size of the outputs of the function. 

---

#### <kbd>property</kbd> statistics

Get all statistics of the function. 



---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/module.py#L192"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `decrypt`

```python
decrypt(
    *results: Union[Value, Tuple[Value, ]]
) → Union[int, ndarray, Tuple[Union[int, ndarray, NoneType], ], NoneType]
```

Decrypt result(s) of evaluation. 



**Args:**
  *results (Union[Value, Tuple[Value, ...]]):  result(s) of evaluation 



**Returns:**
  Optional[Union[int, np.ndarray, Tuple[Optional[Union[int, np.ndarray]], ...]]]:  decrypted result(s) of evaluation 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/module.py#L74"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `draw`

```python
draw(
    horizontal: bool = False,
    save_to: Optional[Path, str] = None,
    show: bool = False
) → Path
```

Draw the graph of the function. 

That this function requires the python `pygraphviz` package which itself requires the installation of `graphviz` packages 

(see https://pygraphviz.github.io/documentation/stable/install.html) 



**Args:**
  horizontal (bool, default = False):  whether to draw horizontally 

 save_to (Optional[Path], default = None):  path to save the drawing  a temporary file will be used if it's None 

 show (bool, default = False):  whether to show the drawing using matplotlib 



**Returns:**
  Path:  path to the drawing 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/module.py#L154"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `encrypt`

```python
encrypt(
    *args: Optional[int, ndarray, List]
) → Union[Value, Tuple[Optional[Value], ], NoneType]
```

Encrypt argument(s) to for evaluation. 



**Args:**
  *args (Optional[Union[int, numpy.ndarray, List]]):  argument(s) for evaluation 



**Returns:**
  Optional[Union[Value, Tuple[Optional[Value], ...]]]:  encrypted argument(s) for evaluation 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/module.py#L210"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `encrypt_run_decrypt`

```python
encrypt_run_decrypt(*args: Any) → Any
```

Encrypt inputs, run the function, and decrypt the outputs in one go. 



**Args:**
  *args (Union[int, numpy.ndarray]):  inputs to the function 



**Returns:**
  Union[int, np.ndarray, Tuple[Union[int, np.ndarray], ...]]:  clear result of homomorphic evaluation 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/module.py#L172"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `run`

```python
run(
    *args: Optional[Value, Tuple[Optional[Value], ]]
) → Union[Value, Tuple[Value, ]]
```

Evaluate the function. 



**Args:**
  *args (Value):  argument(s) for evaluation 



**Returns:**
  Union[Value, Tuple[Value, ...]]:  result(s) of evaluation 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/module.py#L109"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `simulate`

```python
simulate(*args: Any) → Any
```

Simulate execution of the function. 



**Args:**
  *args (Any):  inputs to the function 



**Returns:**
  Any:  result of the simulation 


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/module.py#L514"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FheModule`
Fhe module class, to combine computation graphs, mlir, runtime objects into a single object. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/module.py#L525"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    graphs: Dict[str, Graph],
    mlir: Module,
    compilation_context: CompilationContext,
    configuration: Optional[Configuration] = None
)
```






---

#### <kbd>property</kbd> complexity

Get complexity of the module. 

---

#### <kbd>property</kbd> global_p_error

Get the probability of having at least one simple TLU error during the entire execution. 

---

#### <kbd>property</kbd> keys

Get the keys of the module. 

---

#### <kbd>property</kbd> mlir

Textual representation of the MLIR module. 



**Returns:**
 
 - <b>`str`</b>:  textual representation of the MLIR module 

---

#### <kbd>property</kbd> p_error

Get probability of error for each simple TLU (on a scalar). 

---

#### <kbd>property</kbd> size_of_bootstrap_keys

Get size of the bootstrap keys of the module. 

---

#### <kbd>property</kbd> size_of_keyswitch_keys

Get size of the key switch keys of the module. 

---

#### <kbd>property</kbd> size_of_secret_keys

Get size of the secret keys of the module. 

---

#### <kbd>property</kbd> statistics

Get all statistics of the module. 



---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/module.py#L606"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `cleanup`

```python
cleanup()
```

Cleanup the temporary library output directory. 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/module.py#L673"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `functions`

```python
functions() → Dict[str, FheFunction]
```

Return a dictionnary containing all the functions of the module. 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/module.py#L587"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `keygen`

```python
keygen(
    force: bool = False,
    seed: Optional[int] = None,
    encryption_seed: Optional[int] = None
)
```

Generate keys required for homomorphic evaluation. 



**Args:**
  force (bool, default = False):  whether to generate new keys even if keys are already generated 

 seed (Optional[int], default = None):  seed for private keys randomness 

 encryption_seed (Optional[int], default = None):  seed for encryption randomness 


