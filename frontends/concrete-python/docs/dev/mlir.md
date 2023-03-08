# MLIR

The MLIR project is a sub-project of the LLVM project. It's designed to simplify building domain-specific compilers such as our **Concrete-Compiler**.

**Concrete-Compiler** accepts MLIR as an input and emits compiled assembly code for a target architecture.

**Concrete-Numpy** performs the MLIR generation from the computation graph. Code related to this conversion is in the `concrete/numpy/mlir` folder.

The conversion can be performed using the `convert` method of the `GraphConverter` class.

Within the `convert` method of `GraphConverter`:

* MLIR compatibility of the graph is checked;
* bit width constraints are checked;
* negative lookup tables are offset;
* the computation graph is traversed and each node is converted to their corresponding MLIR representation using the `NodeConverter` class;
* and string representation of the resulting MLIR is returned.
