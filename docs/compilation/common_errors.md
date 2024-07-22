# Common errors

In this document, we list the most common errors, and mention how the user can fix them.

## 1. Could not find a version that satisfies the requirement concrete-python (from versions: none)

**Error message**: `Could not find a version that satisfies the requirement concrete-python (from versions: none)`

**Cause**: The installation does not work fine for you.

**Possible solutions**:
- Be sure that you use a supported Python version (currently from 3.8 to 3.11, included)
- Check you have done `pip install -U pip wheel setuptools` before
- Consider adding a `--extra-index-url https://pypi.zama.ai/cpu` or `--extra-index-url https://pypi.zama.ai/gpu`, depending on whether you
want the CPU or the GPU wheel.
- Concrete requires glibc>=2.28, be sure to have a sufficiently recent version

## 2. Only integers are supported

**Error message**: `RuntimeError: Function you are trying to compile cannot be compiled` with extra information `only integers are supported`

**Cause**: This error can occur if parts of your program contain graphs which are not from integer to integer

**Possible solutions**:
- It is possible to use floats as intermediate values (see the [documentation](../core-features/floating_points.md#floating-points-as-intermediate-values)) but always, inputs and outputs must be integers. So, consider adding ways to convert to integers, such as `.astype(np.uint64)`

## 3. No parameters found

**Error message**: `NoParametersFound`

**Cause**: The optimizer was not able to find cryptographic parameters for the circuit, which are both secure and correct

**Possible solutions**:
- Try to simplify your circuit
- Use smaller weights,
- Add intermediate PBS to reduce the noise, with identity function `fhe.univariate(lambda x: x)`

## 4. Too long inputs for table looup

**Error message**: `RuntimeError: Function you are trying to compile cannot be compiled`, with extra information as `this [...]-bit value is used as an input to a table lookup` with `but only up to 16-bit table lookups are supported`

**Cause**: In your program, you use a table lookup where the input is too large, i.e., is more than 16-bits, which is the current limit

**Possible solutions**:
- Try to simplify your circuit
- Use smaller weights,
- Look to the MLIR to understand where this too-long input comes from

## 5. Impossible to fuse multiple-nodes

**Error message**: `RuntimeError: A subgraph within the function you are trying to compile cannot be fused because it has multiple input nodes`

**Cause**: In your program, you have a subgraph using two nodes or more. It is impossible to fuse such a graph, i.e., to replace it by a table lookup. Concrete will show you where the different nodes are, with some `this is one of the input nodes` printed in the circuit.

**Possible solutions**:
- Try to simplify your circuit
- Have a look to `fhe.multivariate`

## 6. Function is not supported

**Error message**: `RuntimeError: Function '[...]' is not supported`

**Cause**: You are using a function which is not currently supported by Concrete

**Possible solutions**:
- Try to change your program
- Have a look to the documentation to see if there are ways to implement the function differently
- Ask our community channels

## 7. Branching is not allowed

**Error message**: `RuntimeError: Branching within circuits is not possible`

**Cause**: You are using branches in Concrete, it is not allowed in FHE program (typically, if's or
non-constant loops)

**Possible solutions**:
- Change your program
- Consider using tricks to replace ternary-if, as `c ? t : f = f + c * (t-f)`



