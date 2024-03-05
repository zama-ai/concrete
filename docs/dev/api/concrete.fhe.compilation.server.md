<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/server.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.compilation.server`
Declaration of `Server` class. 

**Global Variables**
---------------
- **DEFAULT_GLOBAL_P_ERROR**
- **DEFAULT_P_ERROR**


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/server.py#L51"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Server`
Server class, which can be used to perform homomorphic computation. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/server.py#L68"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    client_specs: ClientSpecs,
    output_dir: Optional[TemporaryDirectory],
    support: LibrarySupport,
    compilation_result: LibraryCompilationResult,
    server_lambda: LibraryLambda,
    is_simulated: bool
)
```






---

#### <kbd>property</kbd> clear_addition_count

Get the number of clear additions in the compiled program. 

---

#### <kbd>property</kbd> clear_addition_count_per_parameter

Get the number of clear additions per parameter in the compiled program. 

---

#### <kbd>property</kbd> clear_addition_count_per_tag

Get the number of clear additions per tag in the compiled program. 

---

#### <kbd>property</kbd> clear_addition_count_per_tag_per_parameter

Get the number of clear additions per tag per parameter in the compiled program. 

---

#### <kbd>property</kbd> clear_multiplication_count

Get the number of clear multiplications in the compiled program. 

---

#### <kbd>property</kbd> clear_multiplication_count_per_parameter

Get the number of clear multiplications per parameter in the compiled program. 

---

#### <kbd>property</kbd> clear_multiplication_count_per_tag

Get the number of clear multiplications per tag in the compiled program. 

---

#### <kbd>property</kbd> clear_multiplication_count_per_tag_per_parameter

Get the number of clear multiplications per tag per parameter in the compiled program. 

---

#### <kbd>property</kbd> complexity

Get complexity of the compiled program. 

---

#### <kbd>property</kbd> encrypted_addition_count

Get the number of encrypted additions in the compiled program. 

---

#### <kbd>property</kbd> encrypted_addition_count_per_parameter

Get the number of encrypted additions per parameter in the compiled program. 

---

#### <kbd>property</kbd> encrypted_addition_count_per_tag

Get the number of encrypted additions per tag in the compiled program. 

---

#### <kbd>property</kbd> encrypted_addition_count_per_tag_per_parameter

Get the number of encrypted additions per tag per parameter in the compiled program. 

---

#### <kbd>property</kbd> encrypted_negation_count

Get the number of encrypted negations in the compiled program. 

---

#### <kbd>property</kbd> encrypted_negation_count_per_parameter

Get the number of encrypted negations per parameter in the compiled program. 

---

#### <kbd>property</kbd> encrypted_negation_count_per_tag

Get the number of encrypted negations per tag in the compiled program. 

---

#### <kbd>property</kbd> encrypted_negation_count_per_tag_per_parameter

Get the number of encrypted negations per tag per parameter in the compiled program. 

---

#### <kbd>property</kbd> global_p_error

Get the probability of having at least one simple TLU error during the entire execution. 

---

#### <kbd>property</kbd> key_switch_count

Get the number of key switches in the compiled program. 

---

#### <kbd>property</kbd> key_switch_count_per_parameter

Get the number of key switches per parameter in the compiled program. 

---

#### <kbd>property</kbd> key_switch_count_per_tag

Get the number of key switches per tag in the compiled program. 

---

#### <kbd>property</kbd> key_switch_count_per_tag_per_parameter

Get the number of key switches per tag per parameter in the compiled program. 

---

#### <kbd>property</kbd> p_error

Get the probability of error for each simple TLU (on a scalar). 

---

#### <kbd>property</kbd> packing_key_switch_count

Get the number of packing key switches in the compiled program. 

---

#### <kbd>property</kbd> packing_key_switch_count_per_parameter

Get the number of packing key switches per parameter in the compiled program. 

---

#### <kbd>property</kbd> packing_key_switch_count_per_tag

Get the number of packing key switches per tag in the compiled program. 

---

#### <kbd>property</kbd> packing_key_switch_count_per_tag_per_parameter

Get the number of packing key switches per tag per parameter in the compiled program. 

---

#### <kbd>property</kbd> programmable_bootstrap_count

Get the number of programmable bootstraps in the compiled program. 

---

#### <kbd>property</kbd> programmable_bootstrap_count_per_parameter

Get the number of programmable bootstraps per parameter in the compiled program. 

---

#### <kbd>property</kbd> programmable_bootstrap_count_per_tag

Get the number of programmable bootstraps per tag in the compiled program. 

---

#### <kbd>property</kbd> programmable_bootstrap_count_per_tag_per_parameter

Get the number of programmable bootstraps per tag per parameter in the compiled program. 

---

#### <kbd>property</kbd> size_of_bootstrap_keys

Get size of the bootstrap keys of the compiled program. 

---

#### <kbd>property</kbd> size_of_inputs

Get size of the inputs of the compiled program. 

---

#### <kbd>property</kbd> size_of_keyswitch_keys

Get size of the key switch keys of the compiled program. 

---

#### <kbd>property</kbd> size_of_outputs

Get size of the outputs of the compiled program. 

---

#### <kbd>property</kbd> size_of_secret_keys

Get size of the secret keys of the compiled program. 

---

#### <kbd>property</kbd> statistics

Get all statistics of the compiled program. 



---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/server.py#L377"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `cleanup`

```python
cleanup()
```

Cleanup the temporary library output directory. 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/server.py#L92"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `create`

```python
create(
    mlir: Union[str, Module],
    configuration: Configuration,
    is_simulated: bool = False,
    compilation_context: Optional[CompilationContext] = None
) → Server
```

Create a server using MLIR and output sign information. 



**Args:**
  mlir (MlirModule):  mlir to compile 

 is_simulated (bool, default = False):  whether to compile in simulation mode or not 

 configuration (Optional[Configuration]):  configuration to use 

 compilation_context (CompilationContext):  context to use for the Compiler 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/server.py#L274"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load`

```python
load(path: Union[str, Path]) → Server
```

Load the server from the given path in zip format. 



**Args:**
  path (Union[str, Path]):  path to load the server from 



**Returns:**
  Server:  server loaded from the filesystem 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/server.py#L322"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `run`

```python
run(
    *args: Optional[Value, Tuple[Optional[Value], ]],
    evaluation_keys: Optional[EvaluationKeys] = None
) → Union[Value, Tuple[Value, ]]
```

Evaluate. 



**Args:**
  *args (Optional[Union[Value, Tuple[Optional[Value], ...]]]):  argument(s) for evaluation 

 evaluation_keys (Optional[EvaluationKeys], default = None):  evaluation keys required for fhe execution 



**Returns:**
  Union[Value, Tuple[Value, ...]]:  result(s) of evaluation 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/server.py#L226"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(path: Union[str, Path], via_mlir: bool = False)
```

Save the server into the given path in zip format. 



**Args:**
  path (Union[str, Path]):  path to save the server 

 via_mlir (bool, default = False):  export using the MLIR code of the program,  this will make the export cross-platform 


