<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/server_circuit.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.compiler.server_circuit`
ServerCircuit. 



---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/server_circuit.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ServerCircuit`
ServerCircuit references a circuit that can be called for execution and simulation. 

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/server_circuit.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(server_circuit: ServerCircuit)
```

Wrap the native Cpp object. 



**Args:**
 
 - <b>`server_circuit`</b> (_ServerCircuit):  object to wrap 



**Raises:**
 
 - <b>`TypeError`</b>:  if server_circuit is not of type _ServerCircuit 




---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/server_circuit.py#L36"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `call`

```python
call(
    public_arguments: PublicArguments,
    evaluation_keys: EvaluationKeys
) → PublicResult
```

Executes the circuit on the public arguments. 



**Args:**
 
 - <b>`public_arguments`</b> (PublicArguments):  public arguments to execute on 
 - <b>`execution_keys`</b> (EvaluationKeys):  evaluation keys to use for execution. 



**Raises:**
 
 - <b>`TypeError`</b>:  if public_arguments is not of type PublicArguments, or if evaluation_keys is  not of type EvaluationKeys 



**Returns:**
 
 - <b>`PublicResult`</b>:  A public result object containing the results. 

---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/compiler/server_circuit.py#L68"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `simulate`

```python
simulate(public_arguments: PublicArguments) → PublicResult
```

Simulates the circuit on the public arguments. 



**Args:**
 
 - <b>`public_arguments`</b> (PublicArguments):  public arguments to execute on 



**Raises:**
 
 - <b>`TypeError`</b>:  if public_arguments is not of type PublicArguments 



**Returns:**
 
 - <b>`PublicResult`</b>:  A public result object containing the results. 


