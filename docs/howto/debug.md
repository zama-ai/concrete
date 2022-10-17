# Debug

In this section, you will learn how to debug the compilation process easily as well as how to get help in case you cannot resolve your issue.

## Debug Artifacts

**Concrete-Numpy** has an artifact system to simplify the process of debugging issues.

### Automatic export.

In case of compilation failures, artifacts are exported automatically to the `.artifacts` directory under the working directory. Let's intentionally create a compilation failure to show what kinds of things are exported.

```python
def f(x):
    return np.sin(x)
```

This function fails to compile because **Concrete-Numpy** does not support floating-point outputs. When you try to compile it, an exception will be raised and the artifacts will be exported automatically. If you go the `.artifacts` directory under the working directory, you'll see the following files:

#### environment.txt

This file contains information about your setup (i.e., your operating system and python version).

```
Linux-5.12.13-arch1-2-x86_64-with-glibc2.29 #1 SMP PREEMPT Fri, 25 Jun 2021 22:56:51 +0000
Python 3.8.10
```

#### requirements.txt

This file contains information about python packages and their versions installed on your system.

```
alabaster==0.7.12
appdirs==1.4.4
argon2-cffi==21.1.0
...
wheel==0.37.0
widgetsnbextension==3.5.1
wrapt==1.12.1
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

#### 1.initial.graph.png

This file contains the visual representation of the initial computation graph right after tracing.

![](../\_static/tutorials/artifacts/auto/1.initial.graph.png)

#### 2.final.graph.txt

This file contains the textual representation of the final computation graph right before MLIR conversion.

```
%0 = x              # EncryptedScalar<uint3>
%1 = sin(%0)        # EncryptedScalar<float64>
return %1
```

#### 2.final.graph.png

This file contains the visual representation of the final computation graph right before MLIR conversion.

![](../\_static/tutorials/artifacts/auto/2.final.graph.png)

#### traceback.txt

This file contains information about the error you received.

```
Traceback (most recent call last):
  File "/home/default/Documents/Projects/Zama/hdk/concrete/numpy/compilation/compiler.py", line 320, in compile
    mlir = GraphConverter.convert(self.graph, virtual=self.configuration.virtual)
  File "/home/default/Documents/Projects/Zama/hdk/concrete/numpy/mlir/graph_converter.py", line 298, in convert
    GraphConverter._check_graph_convertibility(graph)
  File "/home/default/Documents/Projects/Zama/hdk/concrete/numpy/mlir/graph_converter.py", line 175, in _check_graph_convertibility
    raise RuntimeError(
RuntimeError: Function you are trying to compile cannot be converted to MLIR

%0 = x              # EncryptedScalar<uint4>
%1 = sin(%0)        # EncryptedScalar<float64>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ only integer operations are supported
return %1
```

### Manual export.

Manual exports are mostly used for visualization. Nonetheless, they can be very useful for demonstrations. Here is how to perform one:

```python
import concrete.numpy as cnp
import numpy as np

artifacts = cnp.DebugArtifacts("/tmp/custom/export/path")

@cnp.compiler({"x": "encrypted"})
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
%0 = 127                           # ClearScalar<uint7>
%1 = 50                            # ClearScalar<uint6>
%2 = 1                             # ClearScalar<uint1>
%3 = x                             # EncryptedScalar<uint1>
%4 = sin(%3)                       # EncryptedScalar<float64>
%5 = add(%4, %2)                   # EncryptedScalar<float64>
%6 = multiply(%1, %5)              # EncryptedScalar<float64>
%7 = astype(%6, dtype=int_)        # EncryptedScalar<uint1>
%8 = subtract(%0, %7)              # EncryptedScalar<uint1>
return %8
```

#### 1.initial.graph.png

This file contains the visual representation of the initial computation graph right after tracing.

![](../\_static/tutorials/artifacts/manual/1.initial.graph.png)

#### 2.after-float-fuse-0.graph.txt

This file contains the textual representation of the intermediate computation graph after fusing.

```
%0 = 127                     # ClearScalar<uint7>
%1 = x                       # EncryptedScalar<uint1>
%2 = subgraph(%1)            # EncryptedScalar<uint1>
%3 = subtract(%0, %2)        # EncryptedScalar<uint1>
return %3

Subgraphs:

    %2 = subgraph(%1):

        %0 = 50                            # ClearScalar<uint6>
        %1 = 1                             # ClearScalar<uint1>
        %2 = input                         # EncryptedScalar<uint1>
        %3 = sin(%2)                       # EncryptedScalar<float64>
        %4 = add(%3, %1)                   # EncryptedScalar<float64>
        %5 = multiply(%0, %4)              # EncryptedScalar<float64>
        %6 = astype(%5, dtype=int_)        # EncryptedScalar<uint1>
        return %6
```

#### 2.after-fusing.graph.png

This file contains the visual representation of the intermediate computation graph after fusing.

![](../\_static/tutorials/artifacts/manual/2.after-fusing.graph.png)

#### 3.final.graph.txt

This file contains the textual representation of the final computation graph right before MLIR conversion.

```
%0 = 127                     # ClearScalar<uint7>
%1 = x                       # EncryptedScalar<uint3>
%2 = subgraph(%1)            # EncryptedScalar<uint7>
%3 = subtract(%0, %2)        # EncryptedScalar<uint7>
return %3

Subgraphs:

    %2 = subgraph(%1):

        %0 = 50                            # ClearScalar<uint6>
        %1 = 1                             # ClearScalar<uint1>
        %2 = input                         # EncryptedScalar<uint1>
        %3 = sin(%2)                       # EncryptedScalar<float64>
        %4 = add(%3, %1)                   # EncryptedScalar<float64>
        %5 = multiply(%0, %4)              # EncryptedScalar<float64>
        %6 = astype(%5, dtype=int_)        # EncryptedScalar<uint1>
        return %6
```

#### 3.final.graph.png

This file contains the visual representation of the final computation graph right before MLIR conversion.

![](../\_static/tutorials/artifacts/manual/3.final.graph.png)

#### bounds.txt

This file contains information about the bounds of the final computation graph of the function you are compiling using the inputset you provide.

```
%0 :: [127, 127]
%1 :: [0, 7]
%2 :: [2, 95]
%3 :: [32, 125]
```

#### mlir.txt

This file contains information about the MLIR of the function you compiled using the inputset you provided.

```
module  {
  func @main(%arg0: !FHE.eint<7>) -> !FHE.eint<7> {
    %c127_i8 = arith.constant 127 : i8
    %cst = arith.constant dense<"..."> : tensor<128xi64>
    %0 = "FHE.apply_lookup_table"(%arg0, %cst) : (!FHE.eint<7>, tensor<128xi64>) -> !FHE.eint<7>
    %1 = "FHE.sub_int_eint"(%c127_i8, %0) : (i8, !FHE.eint<7>) -> !FHE.eint<7>
    return %1 : !FHE.eint<7>
  }
}
```

## Asking the community

You can seek help with your issue by asking a question directly in the [community forum](https://community.zama.ai/).

## Submitting an issue

If you cannot find a solution in the community forum, or you found a bug in the library, you could create an issue in our GitHub repository.

In case of a bug:

* try to minimize randomness
* try to minimize your function as much as possible while keeping the bug - this will help to fix the bug faster
* try to include your inputset in the issue
* try to include reproduction steps in the issue
* try to include debug artifacts in the issue

In case of a feature request:

* try to give a minimal example of the desired behavior
* try to explain your use case
