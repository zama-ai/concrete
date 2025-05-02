<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/compilation/circuit.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.compilation.circuit`
Declaration of `Circuit` class. 



---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/circuit.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Circuit`
Circuit class, to combine computation graph, mlir, client and server into a single object. 

<a href="../../frontends/concrete-python/concrete/fhe/compilation/circuit.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(module: FheModule)
```






---

#### <kbd>property</kbd> clear_addition_count

Get the number of clear additions in the circuit. 

---

#### <kbd>property</kbd> clear_addition_count_per_parameter

Get the number of clear additions per parameter in the circuit. 

---

#### <kbd>property</kbd> clear_addition_count_per_tag

Get the number of clear additions per tag in the circuit. 

---

#### <kbd>property</kbd> clear_addition_count_per_tag_per_parameter

Get the number of clear additions per tag per parameter in the circuit. 

---

#### <kbd>property</kbd> clear_multiplication_count

Get the number of clear multiplications in the circuit. 

---

#### <kbd>property</kbd> clear_multiplication_count_per_parameter

Get the number of clear multiplications per parameter in the circuit. 

---

#### <kbd>property</kbd> clear_multiplication_count_per_tag

Get the number of clear multiplications per tag in the circuit. 

---

#### <kbd>property</kbd> clear_multiplication_count_per_tag_per_parameter

Get the number of clear multiplications per tag per parameter in the circuit. 

---

#### <kbd>property</kbd> client

Return the circuit client. 

---

#### <kbd>property</kbd> compilation_context

Return the circuit compilation context. 

---

#### <kbd>property</kbd> complexity

Get complexity of the circuit. 

---

#### <kbd>property</kbd> configuration

Return the circuit configuration. 

---

#### <kbd>property</kbd> encrypted_addition_count

Get the number of encrypted additions in the circuit. 

---

#### <kbd>property</kbd> encrypted_addition_count_per_parameter

Get the number of encrypted additions per parameter in the circuit. 

---

#### <kbd>property</kbd> encrypted_addition_count_per_tag

Get the number of encrypted additions per tag in the circuit. 

---

#### <kbd>property</kbd> encrypted_addition_count_per_tag_per_parameter

Get the number of encrypted additions per tag per parameter in the circuit. 

---

#### <kbd>property</kbd> encrypted_negation_count

Get the number of encrypted negations in the circuit. 

---

#### <kbd>property</kbd> encrypted_negation_count_per_parameter

Get the number of encrypted negations per parameter in the circuit. 

---

#### <kbd>property</kbd> encrypted_negation_count_per_tag

Get the number of encrypted negations per tag in the circuit. 

---

#### <kbd>property</kbd> encrypted_negation_count_per_tag_per_parameter

Get the number of encrypted negations per tag per parameter in the circuit. 

---

#### <kbd>property</kbd> global_p_error

Get the probability of having at least one simple TLU error during the entire execution. 

---

#### <kbd>property</kbd> graph

Return the circuit graph. 

---

#### <kbd>property</kbd> key_switch_count

Get the number of key switches in the circuit. 

---

#### <kbd>property</kbd> key_switch_count_per_parameter

Get the number of key switches per parameter in the circuit. 

---

#### <kbd>property</kbd> key_switch_count_per_tag

Get the number of key switches per tag in the circuit. 

---

#### <kbd>property</kbd> key_switch_count_per_tag_per_parameter

Get the number of key switches per tag per parameter in the circuit. 

---

#### <kbd>property</kbd> keys

Get the keys of the circuit. 

---

#### <kbd>property</kbd> memory_usage_per_location

Get the memory usage of operations in the circuit per location. 

---

#### <kbd>property</kbd> mlir

Textual representation of the MLIR module. 



**Returns:**
 
 - <b>`str`</b>:  textual representation of the MLIR module 

---

#### <kbd>property</kbd> mlir_module

Return the circuit mlir module. 

---

#### <kbd>property</kbd> p_error

Get probability of error for each simple TLU (on a scalar). 

---

#### <kbd>property</kbd> packing_key_switch_count

Get the number of packing key switches in the circuit. 

---

#### <kbd>property</kbd> packing_key_switch_count_per_parameter

Get the number of packing key switches per parameter in the circuit. 

---

#### <kbd>property</kbd> packing_key_switch_count_per_tag

Get the number of packing key switches per tag in the circuit. 

---

#### <kbd>property</kbd> packing_key_switch_count_per_tag_per_parameter

Get the number of packing key switches per tag per parameter in the circuit. 

---

#### <kbd>property</kbd> programmable_bootstrap_count

Get the number of programmable bootstraps in the circuit. 

---

#### <kbd>property</kbd> programmable_bootstrap_count_per_parameter

Get the number of programmable bootstraps per bit width in the circuit. 

---

#### <kbd>property</kbd> programmable_bootstrap_count_per_tag

Get the number of programmable bootstraps per tag in the circuit. 

---

#### <kbd>property</kbd> programmable_bootstrap_count_per_tag_per_parameter

Get the number of programmable bootstraps per tag per bit width in the circuit. 

---

#### <kbd>property</kbd> server

Return the circuit server. 

---

#### <kbd>property</kbd> simulator

Return the circuit simulator. 

---

#### <kbd>property</kbd> size_of_bootstrap_keys

Get size of the bootstrap keys of the circuit. 

---

#### <kbd>property</kbd> size_of_inputs

Get size of the inputs of the circuit. 

---

#### <kbd>property</kbd> size_of_keyswitch_keys

Get size of the key switch keys of the circuit. 

---

#### <kbd>property</kbd> size_of_outputs

Get size of the outputs of the circuit. 

---

#### <kbd>property</kbd> size_of_secret_keys

Get size of the secret keys of the circuit. 

---

#### <kbd>property</kbd> statistics

Get all statistics of the circuit. 



---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/circuit.py#L226"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `cleanup`

```python
cleanup()
```

Cleanup the temporary library output directory. 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/circuit.py#L193"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `decrypt`

```python
decrypt(
    *results: Union[Value, tuple[Value, ]]
) → Union[int, ndarray, tuple[Union[int, ndarray, NoneType], ], NoneType]
```

Decrypt result(s) of evaluation. 



**Args:**
  *results (Union[Value, Tuple[Value, ...]]):  result(s) of evaluation 



**Returns:**
  Optional[Union[int, np.ndarray, Tuple[Optional[Union[int, np.ndarray]], ...]]]:  decrypted result(s) of evaluation 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/circuit.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `draw`

```python
draw(
    horizontal: bool = False,
    save_to: Optional[Path, str] = None,
    show: bool = False
) → Path
```

Draw the graph of the circuit. 

That this function requires the python `pygraphviz` package which itself requires the installation of `graphviz` packages 

(see https://pygraphviz.github.io/documentation/stable/install.html) 



**Args:**
  horizontal (bool, default = False):  whether to draw horizontally 

 save_to (Optional[Path], default = None):  path to save the drawing  a temporary file will be used if it's None 

 show (bool, default = False):  whether to show the drawing using matplotlib 



**Returns:**
  Path:  path to the drawing 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/circuit.py#L93"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `enable_fhe_execution`

```python
enable_fhe_execution()
```

Enable FHE execution. 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/circuit.py#L87"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `enable_fhe_simulation`

```python
enable_fhe_simulation()
```

Enable FHE simulation. 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/circuit.py#L158"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `encrypt`

```python
encrypt(
    *args: Optional[int, ndarray, list]
) → Union[Value, tuple[Optional[Value], ], NoneType]
```

Encrypt argument(s) to for evaluation. 



**Args:**
  *args (Optional[Union[int, numpy.ndarray, List]]):  argument(s) for evaluation 



**Returns:**
  Optional[Union[Value, Tuple[Optional[Value], ...]]]:  encrypted argument(s) for evaluation 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/circuit.py#L211"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `encrypt_run_decrypt`

```python
encrypt_run_decrypt(*args: Any) → Any
```

Encrypt inputs, run the circuit, and decrypt the outputs in one go. 



**Args:**
  *args (Union[int, numpy.ndarray]):  inputs to the circuit 



**Returns:**
  Union[int, np.ndarray, Tuple[Union[int, np.ndarray], ...]]:  clear result of homomorphic evaluation 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/circuit.py#L128"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `keygen`

```python
keygen(
    force: bool = False,
    seed: Optional[int] = None,
    encryption_seed: Optional[int] = None,
    initial_keys: Optional[dict[int, LweSecretKey]] = None
)
```

Generate keys required for homomorphic evaluation. 



**Args:**
  force (bool, default = False):  whether to generate new keys even if keys are already generated 

 seed (Optional[int], default = None):  seed for private keys randomness 

 encryption_seed (Optional[int], default = None):  seed for encryption randomness 

 initial_keys (Optional[Dict[int, LweSecretKey]] = None):  initial keys to set before keygen 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/circuit.py#L175"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `run`

```python
run(
    *args: Optional[Value, tuple[Optional[Value], ]]
) → Union[Value, tuple[Value, ]]
```

Evaluate the circuit. 



**Args:**
  *args (Value):  argument(s) for evaluation 



**Returns:**
  Union[Value, Tuple[Value, ...]]:  result(s) of evaluation 

---

<a href="../../frontends/concrete-python/concrete/fhe/compilation/circuit.py#L99"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `simulate`

```python
simulate(*args: Any) → Any
```

Simulate execution of the circuit. 



**Args:**
  *args (Any):  inputs to the circuit 



**Returns:**
  Any:  result of the simulation 


