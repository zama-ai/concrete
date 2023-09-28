# Debug

In this section, you will learn how to debug the compilation process easily and find help in the case that you cannot resolve your issue.

## Compiler debug and verbose modes

There are two [configuration](../howto/configure.md) options that you can use to understand what's happening under the hood during compilation process.

- **compiler_verbose_mode** will print the passes applied by the compiler and let you see the transformations done by the compiler. Also, in case of crashes, it could narrow down the crash location. 
- **compiler_debug_mode** is a lot more detailed version of the verbose mode. Even better for crashes.

{% hint style="warning" %}
These flags might not work as expected in Jupyter notebooks as they output to stderr directly from C++.
{% endhint %}

## Debug artifacts

**Concrete** has an artifact system to simplify the process of debugging issues.

### Automatic export.

In case of compilation failures, artifacts are exported automatically to the `.artifacts` directory under the working directory. Let's intentionally create a compilation failure to show what is exported.

```python
def f(x):
    return np.sin(x)
```

This function fails to compile because **Concrete** does not support floating-point outputs. When you try to compile it, an exception will be raised and the artifacts will be exported automatically. If you go to the `.artifacts` directory under the working directory, you'll see the following files:

#### environment.txt

This file contains information about your setup (i.e., your operating system and python version).

```
Linux-5.12.13-arch1-2-x86_64-with-glibc2.29 #1 SMP PREEMPT Fri, 25 Jun 2021 22:56:51 +0000
Python 3.8.10
```

#### requirements.txt

This file contains information about Python packages and their versions installed on your system.

```
astroid==2.15.0
attrs==22.2.0
auditwheel==5.3.0
...
wheel==0.40.0
wrapt==1.15.0
zipp==3.15.0
```

#### function.txt

This file contains information about the function you tried to compile.

```
def f(x):
    return np.sin(x)
```

#### parameters.txt

This file contains information about the encryption status of the parameters of the function you tried to compile.

```
x :: encrypted
```

#### 1.initial.graph.txt

This file contains the textual representation of the initial computation graph right after tracing.

```
%0 = x              # EncryptedScalar<uint3>
%1 = sin(%0)        # EncryptedScalar<float64>
return %1
```

#### 2.final.graph.txt

This file contains the textual representation of the final computation graph right before MLIR conversion.

```
%0 = x              # EncryptedScalar<uint3>
%1 = sin(%0)        # EncryptedScalar<float64>
return %1
```

#### traceback.txt

This file contains information about the error you received.

```
Traceback (most recent call last):
  File "/path/to/your/script.py", line 9, in <module>
    circuit = f.compile(inputset)
  File "/usr/local/lib/python3.10/site-packages/concrete/fhe/compilation/decorators.py", line 159, in compile
    return self.compiler.compile(inputset, configuration, artifacts, **kwargs)
  File "/usr/local/lib/python3.10/site-packages/concrete/fhe/compilation/compiler.py", line 437, in compile
    mlir = GraphConverter.convert(self.graph)
  File "/usr/local/lib/python3.10/site-packages/concrete/fhe/mlir/graph_converter.py", line 677, in convert
    GraphConverter._check_graph_convertibility(graph)
  File "/usr/local/lib/python3.10/site-packages/concrete/fhe/mlir/graph_converter.py", line 240, in _check_graph_convertibility
    raise RuntimeError(message)
RuntimeError: Function you are trying to compile cannot be converted to MLIR

%0 = x              # EncryptedScalar<uint3>          ∈ [3, 5]
%1 = sin(%0)        # EncryptedScalar<float64>        ∈ [-0.958924, 0.14112]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only integer operations are supported
                                                                             /path/to/your/script.py:6
return %1
```

### Manual exports.

Manual exports are mostly used for visualization. They can be very useful for demonstrations. Here is how to perform one:

```python
from concrete import fhe
import numpy as np

artifacts = fhe.DebugArtifacts("/tmp/custom/export/path")

@fhe.compiler({"x": "encrypted"})
def f(x):
    return 127 - (50 * (np.sin(x) + 1)).astype(np.int64)

inputset = range(2 ** 3)
circuit = f.compile(inputset, artifacts=artifacts)

artifacts.export()
```

If you go to the `/tmp/custom/export/path` directory, you'll see the following files:

#### 1.initial.graph.txt

This file contains the textual representation of the initial computation graph right after tracing.

```
%0 = x                             # EncryptedScalar<uint1>
%1 = sin(%0)                       # EncryptedScalar<float64>
%2 = 1                             # ClearScalar<uint1>
%3 = add(%1, %2)                   # EncryptedScalar<float64>
%4 = 50                            # ClearScalar<uint6>
%5 = multiply(%4, %3)              # EncryptedScalar<float64>
%6 = astype(%5, dtype=int_)        # EncryptedScalar<uint1>
%7 = 127                           # ClearScalar<uint7>
%8 = subtract(%7, %6)              # EncryptedScalar<uint1>
return %8
```

#### 2.after-fusing.graph.txt

This file contains the textual representation of the intermediate computation graph after fusing.

```
%0 = x                       # EncryptedScalar<uint1>
%1 = subgraph(%0)            # EncryptedScalar<uint1>
%2 = 127                     # ClearScalar<uint7>
%3 = subtract(%2, %1)        # EncryptedScalar<uint1>
return %3

Subgraphs:

    %1 = subgraph(%0):

        %0 = input                         # EncryptedScalar<uint1>
        %1 = sin(%0)                       # EncryptedScalar<float64>
        %2 = 1                             # ClearScalar<uint1>
        %3 = add(%1, %2)                   # EncryptedScalar<float64>
        %4 = 50                            # ClearScalar<uint6>
        %5 = multiply(%4, %3)              # EncryptedScalar<float64>
        %6 = astype(%5, dtype=int_)        # EncryptedScalar<uint1>
        return %6
```

#### 3.final.graph.txt

This file contains the textual representation of the final computation graph right before MLIR conversion.

```
%0 = x                       # EncryptedScalar<uint3>        ∈ [0, 7]
%1 = subgraph(%0)            # EncryptedScalar<uint7>        ∈ [2, 95]
%2 = 127                     # ClearScalar<uint7>            ∈ [127, 127]
%3 = subtract(%2, %1)        # EncryptedScalar<uint7>        ∈ [32, 125]
return %3

Subgraphs:

    %1 = subgraph(%0):

        %0 = input                         # EncryptedScalar<uint1>
        %1 = sin(%0)                       # EncryptedScalar<float64>
        %2 = 1                             # ClearScalar<uint1>
        %3 = add(%1, %2)                   # EncryptedScalar<float64>
        %4 = 50                            # ClearScalar<uint6>
        %5 = multiply(%4, %3)              # EncryptedScalar<float64>
        %6 = astype(%5, dtype=int_)        # EncryptedScalar<uint1>
        return %6
```

#### mlir.txt

This file contains information about the MLIR of the function you compiled using the inputset you provided.

```
module {
  func.func @main(%arg0: !FHE.eint<7>) -> !FHE.eint<7> {
    %c127_i8 = arith.constant 127 : i8
    %cst = arith.constant dense<"..."> : tensor<128xi64>
    %0 = "FHE.apply_lookup_table"(%arg0, %cst) : (!FHE.eint<7>, tensor<128xi64>) -> !FHE.eint<7>
    %1 = "FHE.sub_int_eint"(%c127_i8, %0) : (i8, !FHE.eint<7>) -> !FHE.eint<7>
    return %1 : !FHE.eint<7>
  }
}
```

#### client\_parameters.json

This file contains information about the client parameters chosen by **Concrete**.

```
{
    "bootstrapKeys": [
        {
            "baseLog": 22,
            "glweDimension": 1,
            "inputLweDimension": 908,
            "inputSecretKeyID": 1,
            "level": 1,
            "outputSecretKeyID": 0,
            "polynomialSize": 8192,
            "variance": 4.70197740328915e-38
        }
    ],
    "functionName": "main",
    "inputs": [
        {
            "encryption": {
                "encoding": {
                    "isSigned": false,
                    "precision": 7
                },
                "secretKeyID": 0,
                "variance": 4.70197740328915e-38
            },
            "shape": {
                "dimensions": [],
                "sign": false,
                "size": 0,
                "width": 7
            }
        }
    ],
    "keyswitchKeys": [
        {
            "baseLog": 3,
            "inputSecretKeyID": 0,
            "level": 6,
            "outputSecretKeyID": 1,
            "variance": 1.7944329123150665e-13
        }
    ],
    "outputs": [
        {
            "encryption": {
                "encoding": {
                    "isSigned": false,
                    "precision": 7
                },
                "secretKeyID": 0,
                "variance": 4.70197740328915e-38
            },
            "shape": {
                "dimensions": [],
                "sign": false,
                "size": 0,
                "width": 7
            }
        }
    ],
    "packingKeyswitchKeys": [],
    "secretKeys": [
        {
            "dimension": 8192
        },
        {
            "dimension": 908
        }
    ]
}
```

## Asking the community

You can seek help with your issue by asking a question directly in the [community forum](https://community.zama.ai/).

## Submitting an issue

If you cannot find a solution in the community forum, or you found a bug in the library, you could create an issue in our GitHub repository.

In case of a bug, try to:

* minimize randomness;
* minimize your function as much as possible while keeping the bug - this will help to fix the bug faster;
* include your inputset in the issue;
* include reproduction steps in the issue;
* include debug artifacts in the issue.

In case of a feature request, try to:

* give a minimal example of the desired behavior;
* explain your use case.
