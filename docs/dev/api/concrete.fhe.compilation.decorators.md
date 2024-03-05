<!-- markdownlint-disable -->

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/decorators.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.compilation.decorators`
Declaration of `circuit` and `compiler` decorators. 


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/decorators.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `circuit`

```python
circuit(
    parameters: Mapping[str, Union[str, EncryptionStatus]],
    configuration: Optional[Configuration] = None,
    artifacts: Optional[DebugArtifacts] = None,
    **kwargs
)
```

Provide a direct interface for compilation. 



**Args:**
  parameters (Mapping[str, Union[str, EncryptionStatus]]):  encryption statuses of the parameters of the function to compile 

 configuration(Optional[Configuration], default = None):  configuration to use 

 artifacts (Optional[DebugArtifacts], default = None):  artifacts to store information about the process 

 kwargs (Dict[str, Any]):  configuration options to overwrite 


---

<a href="../../../compilers/concrete-compiler/compiler/lib/Bindings/Python/concrete/fhe/compilation/decorators.py#L78"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compiler`

```python
compiler(parameters: Mapping[str, Union[str, EncryptionStatus]])
```

Provide an easy interface for compilation. 



**Args:**
  parameters (Mapping[str, Union[str, EncryptionStatus]]):  encryption statuses of the parameters of the function to compile 


