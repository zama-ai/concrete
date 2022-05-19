# MLIR

The MLIR project is a sub-project of the LLVM project. It's designed to simplify building domain-specific compilers such as ours: Concrete Compiler.

Concrete Compiler accepts MLIR as input and emits compiled assembly code for the target architecture.

Concrete NumPy does the MLIR generation from the computation graph. Code related to this conversion is in `concrete/numpy/mlir` folder.

The conversion can be performed using `convert` method of `GraphConverter` class.

Within `convert` method of `GraphConverter`:

* MLIR compatibility of the graph is checked
* Bit-width constraints are checked
* Negative lookup tables are offsetted
* Computation graph is traversed and each node is converted to their corresponding MLIR representation using `NodeConverter` class
* String representation of resulting MLIR is returned
