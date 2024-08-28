### Enabling dataflow parallelism

This guide explains dataflow parallelism and how it can improve the execution time of **Concrete** circuits.

Dataflow parallelism is particularly useful when the circuit performs computations that are neither completely independent (such as loop/doall parallelism) nor fully dependent (e.g. sequential, non-parallelizable code). In such cases dataflow tasks can execute as soon as their inputs are available and thus minimizing over-synchronization.

Without dataflow parallelism, circuit is executed operation by operation, like an imperative language. If the operations themselves are not tensorized, loop parallelism would not be utilized and the entire execution would happen in a single thread. Dataflow parallelism changes this by analyzing the operations and their dependencies within the circuit to determine what can be done in parallel and what cannot. Then it distributes the tasks that can be done in parallel to different threads.

For example:

```python
import time

import numpy as np
from concrete import fhe

def f(x, y, z):
    # normally, you'd use fhe.array to construct a concrete tensor
    # but for this example, we just create a simple numpy array
    # so the matrix multiplication can happen on a cellular level
    a = np.array([[x, y], [z, 2]])
    b = np.array([[1, x], [z, y]])
    return fhe.array(a @ b)

inputset = fhe.inputset(fhe.uint3, fhe.uint3, fhe.uint3)

for dataflow_parallelize in [False, True]:
    compiler = fhe.Compiler(f, {"x": "encrypted", "y": "encrypted", "z": "encrypted"})
    circuit = compiler.compile(inputset, dataflow_parallelize=dataflow_parallelize)

    circuit.keygen()
    for sample in inputset[:3]:  # warmup
        circuit.encrypt_run_decrypt(*sample)

    timings = []
    for sample in inputset[3:13]:
        start = time.time()
        result = circuit.encrypt_run_decrypt(*sample)
        end = time.time()

        assert np.array_equal(result, f(*sample))
        timings.append(end - start)

    if not dataflow_parallelize:
        print(f"without dataflow parallelize -> {np.mean(timings):.03f}s")
    else:
        print(f"   with dataflow parallelize -> {np.mean(timings):.03f}s")
```

This prints:

```
without dataflow parallelize -> 0.609s
   with dataflow parallelize -> 0.414s
```

The reason for that is:

```
// this is the generated MLIR for the circuit
// without dataflow, every single line would be executed one after the other

module {
  func.func @main(%arg0: !FHE.eint<7>, %arg1: !FHE.eint<7>, %arg2: !FHE.eint<7>) -> tensor<2x2x!FHE.eint<7>> {
  
    // but if you look closely, you can see that this multiplication
    %c1_i2 = arith.constant 1 : i2
    %0 = "FHE.mul_eint_int"(%arg0, %c1_i2) : (!FHE.eint<7>, i2) -> !FHE.eint<7>
    
    // is completely independent of this one, so dataflow makes them run in parallel
    %1 = "FHE.mul_eint"(%arg1, %arg2) : (!FHE.eint<7>, !FHE.eint<7>) -> !FHE.eint<7>
    
    // however, this addition needs the first two operations
    // so dataflow waits until both are done before performing this one
    %2 = "FHE.add_eint"(%0, %1) : (!FHE.eint<7>, !FHE.eint<7>) -> !FHE.eint<7>
    
    // lastly, this multiplication is completely independent from the first three operations
    // so its execution starts in parallel when execution starts with dataflow
    %3 = "FHE.mul_eint"(%arg0, %arg0) : (!FHE.eint<7>, !FHE.eint<7>) -> !FHE.eint<7>
    
    // similar logic can be applied to the remaining operations...
    %4 = "FHE.mul_eint"(%arg1, %arg1) : (!FHE.eint<7>, !FHE.eint<7>) -> !FHE.eint<7>
    %5 = "FHE.add_eint"(%3, %4) : (!FHE.eint<7>, !FHE.eint<7>) -> !FHE.eint<7>
    %6 = "FHE.mul_eint_int"(%arg2, %c1_i2) : (!FHE.eint<7>, i2) -> !FHE.eint<7>
    %c2_i3 = arith.constant 2 : i3
    %7 = "FHE.mul_eint_int"(%arg2, %c2_i3) : (!FHE.eint<7>, i3) -> !FHE.eint<7>
    %8 = "FHE.add_eint"(%6, %7) : (!FHE.eint<7>, !FHE.eint<7>) -> !FHE.eint<7>
    %9 = "FHE.mul_eint"(%arg2, %arg0) : (!FHE.eint<7>, !FHE.eint<7>) -> !FHE.eint<7>
    %10 = "FHE.mul_eint_int"(%arg1, %c2_i3) : (!FHE.eint<7>, i3) -> !FHE.eint<7>
    %11 = "FHE.add_eint"(%9, %10) : (!FHE.eint<7>, !FHE.eint<7>) -> !FHE.eint<7>
    %from_elements = tensor.from_elements %2, %5, %8, %11 : tensor<2x2x!FHE.eint<7>>
    return %from_elements : tensor<2x2x!FHE.eint<7>>
    
  }
}
```

To summarize, dataflow analyzes the circuit to determine which parts of the circuit can be run at the same time, and tries to run as many operations as possible in parallel.

{% hint style="warning" %}
When the circuit is tensorized, dataflow might slow execution down since the tensor operations already use multiple threads and adding dataflow on top creates congestion in the CPU between the HPX (dataflow parallelism runtime) and OpenMP (loop parallelism runtime). So try both before deciding on whether to use dataflow or not.
{% endhint %}
