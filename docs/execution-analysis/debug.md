# Debug

This document provides guidance on debugging the compilation process.

## Compiler debug and verbose modes

Two [configuration](../guides/configure.md) options are available to help you understand the compilation process:

* **compiler\_verbose\_mode**: Prints the compiler passes and shows the transformations applied. It can help identify the crash location if a crash occurs.
* **compiler\_debug\_mode**: A more detailed version of the verbose mode, providing additional information, particularly useful for diagnosing crashes.

{% hint style="warning" %}
These flags might not work as expected in Jupyter notebooks as they output to `stderr` directly from C++.
{% endhint %}

## Debug artifacts

**Concrete** includes an artifact system that simplifies the debugging process by automatically or manually exporting detailed information during compilation failures.

### Automatic export
When a compilation fails, artifacts are automatically exported to the `.artifacts` directory in the working directory. Here's an example of what gets exported when a function fails to compile:

```python
def f(x):
    return np.sin(x)
```

This function fails to compile because **Concrete** does not support floating-point outputs. When you try to compile it, an exception will be raised and the artifacts will be exported automatically. The following files will be generated in the `.artifacts` directory:

- **`environment.txt`**: Information about your system setup, including the operating system and Python version.

```
Linux-5.12.13-arch1-2-x86_64-with-glibc2.29 #1 SMP PREEMPT Fri, 25 Jun 2021 22:56:51 +0000
Python 3.8.10
```

- **`requirements.txt`**: The installed Python packages and their versions.

```
astroid==2.15.0
attrs==22.2.0
auditwheel==5.3.0
...
wheel==0.40.0
wrapt==1.15.0
zipp==3.15.0
```

- **`function.txt`**: The code of the function that failed to compile.

```
def f(x):
    return np.sin(x)
```

- **`parameters.txt`**: Information about the encryption status function's parameters.

```
x :: encrypted
```

- **`1.initial.graph.txt`**: The textual representation of the initial computation graph right after tracing.

```
%0 = x              # EncryptedScalar<uint3>
%1 = sin(%0)        # EncryptedScalar<float64>
return %1
```

- **`final.graph.txt`**: The textual representation of the final computation graph right before MLIR conversion.

```
%0 = x              # EncryptedScalar<uint3>
%1 = sin(%0)        # EncryptedScalar<float64>
return %1
```

- **`traceback.txt`**: Details of the error occurred.

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

### Manual exports

Manual exports are mostly used for visualization and demonstrations. Here is how to perform one:

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

After running the code, you will find the following files under `/tmp/custom/export/path` directory: 

- **`1.initial.graph.txt`**: The textual representation of the initial computation graph right after tracing.

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

- **`2.after-fusing.graph.txt`**: The textual representation of the intermediate computation graph after fusing.

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

- **`3.final.graph.txt`**: The textual representation of the final computation graph right before MLIR conversion.

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

- **`mlir.txt`**: Information about the MLIR of the function which was compiled using the provided input-set.

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

- **`client\_parameters.json`**: Information about the client parameters chosen by **Concrete**.

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

If you cannot find a solution in the community forum, or if you have found a bug in the library, you could [create an issue](https://github.com/zama-ai/concrete/issues/new/choose) in our GitHub repository.

For [bug reports](https://github.com/zama-ai/concrete/issues/new?assignees=&labels=bug%2C+triage&projects=&template=bug_report.md), try to:

* Avoid randomness to ensure reproducibility of the bug
* Minimize your function while keeping the bug to expedite the fix
* Include your input-set in the issue
* Provide clear reproduction steps
* Include debug artifacts in the issue

For [feature requests](https://github.com/zama-ai/concrete/issues/new?assignees=&labels=feature&projects=&template=features.md), try to:

* Give a minimal example of the desired behavior
* Explain your use case
