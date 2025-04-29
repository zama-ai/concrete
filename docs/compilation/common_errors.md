# Common errors

This document explains the most common errors and provides solutions to fix them.

## 1. Could not find a version that satisfies the requirement concrete-python (from versions: none)

**Error message**: `Could not find a version that satisfies the requirement concrete-python (from versions: none)`

**Cause**: The installation does not work fine for you.

**Possible solutions**:
- Be sure that you use a supported Python version (currently from 3.9 to 3.12, included).
- Check that you have done `pip install -U pip wheel setuptools` before.
- Consider adding a `--extra-index-url https://pypi.zama.ai/cpu` or `--extra-index-url https://pypi.zama.ai/gpu`, depending on whether you want the CPU or the GPU wheel.
- Concrete requires glibc>=2.28, be sure to have a sufficiently recent version.

## 2. Only integers are supported

**Error message**: `RuntimeError: Function you are trying to compile cannot be compiled` with extra information `only integers are supported`

**Cause**: Parts of your program contain graphs that are not from integer to integer

**Possible solutions**:
- You can use floats as intermediate values (see the [documentation](../core-features/floating_points.md#floating-points-as-intermediate-values)). However, both inputs and outputs must be integers. Consider converting values to integers, such as `.astype(np.uint64)`

## 3. No parameters found

**Error message**: `NoParametersFound`

**Cause**: The optimizer can't find cryptographic parameters for the circuit that are both secure and correct.

**Possible solutions**:
- Try to simplify your circuit.
- Use smaller weights.
- Add intermediate PBS to reduce the noise, with identity function `fhe.refresh(lambda x: x)`.

## 4. Too long inputs for table looup

**Error message**: `RuntimeError: Function you are trying to compile cannot be compiled`, with extra information as `this [...]-bit value is used as an input to a table lookup` with `but only up to 16-bit table lookups are supported`

**Cause**: The program uses a Table Lookup that contains oversized inputs exceeding the current 16-bit limit.

**Possible solutions**:
- Try to simplify your circuit.
- Use smaller weights.
- Look to the graph to understand where this oversized input comes from and ensure that the input size for Table Lookup operations does not exceed 16 bits.
- Use `show_bit_width_constraints=True` to understand bit widths are assigned the way they are.

## 5. Impossible to fuse multiple-nodes

**Error message**: `RuntimeError: A subgraph within the function you are trying to compile cannot be fused because it has multiple input nodes`

**Cause**: A subgraph in your program uses two or more input nodes. It is impossible to fuse such a graph, meaning replace it by a table lookup. Concrete will indicate the input nodes with `this is one of the input nodes printed` in the circuit.

**Possible solutions**:
- Try to simplify your circuit.
- Have a look to `fhe.multivariate`.

## 6. Function is not supported

**Error message**: `RuntimeError: Function '[...]' is not supported`

**Cause**: The function used is not currently supported by Concrete.

**Possible solutions**:
- Try to change your program.
- Check the corresponding documentation to see if there are ways to implement the function differently.
- Post your issue in our [community channels](https://community.zama.ai/c/concrete/7).

## 7. Branching is not allowed

**Error message**: `RuntimeError: Branching within circuits is not possible`

**Cause**: Branching operations, such as if statements or non-constant loops, are not supported in Concrete's FHE programs.

**Possible solutions**:
- Change your program.
- Consider using tricks to replace ternary-if, as `c ? t : f = f + c * (t-f)`.

## 8. Unfeasible noise constraint

**Error message**: `Unfeasible noise constraint encountered`

**Cause**: The optimizer can't find cryptographic parameters for the circuit that are both secure and correct.

**Possible solutions**:
- Try to simplify your circuit.
- Use smaller weights.
- Add intermediate PBS to reduce the noise, with identity function `fhe.refresh(x)`.

## 9. Non composable circuit

**Error message**: `Program can not be composed`

**Cause**: Some circuit outputs are contaminated by unrefreshed input noise.

**Possible solutions**:
- Add intermediate PBS to refresh the noise with `fhe.refresh(x)`.
