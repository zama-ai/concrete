# Quick overview

In this document, we give a quick overview of the philosophy behind Concrete.

## Functions

### Available FHE-friendly functions

Concrete is a compiler, which aims to turn Python code into its FHE equivalent, in a process which is
called the FHE compilation. The best efforts were made to simplify the process: in particular,
exceptions apart, the same functions than the Python users are used to use are available. More complete
list of available functions is given [in the reference section](../dev/compatibility.md).

### Levelled vs non-levelled operations

Basically, in the compiled circuit, there will be two kind of operations:
- levelled operations, which are the additions, subtractions or multiplications by a constant; these
operations are also called the linear operations
- Table Lookup (TLU) operations, which are used to do anything which is not linear.

TLU operations are essential to be able to compile complex functions. We explain their use in
different sections of the documentation: [direct TLU use](../core-features/table_lookups.md) or
[internal use to replace some non-linear functions](../core-features/non_linear_operations.md). We have
tools in Concrete to replace univariate or multivariate non-linear functions (ie, functions of one
or more inputs) by TLU.

TLU are more costly that levelled operations, so we also explain how to limit their impact.

Remark that matrix multiplication (aka Gemm -- General Matrix multiplication) and convolutions are
levelled operations, since they imply only additions and multiplications by constant.

### Conditional branches and loops

Functions can't use conditional branches or non-constant-size loops,
unless [modules](../compilation/composing_functions_with_modules.md) are used. However,
control flow statements with constant values are allowed, for example,
`for i in range(SOME_CONSTANT)`, `if os.environ.get("SOME_FEATURE") == "ON":`.

## Data

### Integers

In Concrete, everything needs to be an integer. Users needing floats can quantize to integers before
encryption, operate on integers and dequantize to floats after decryption: all of this is done for
the user in Concrete ML. However, you can have floating-point intermediate values as long as they can
be converted to an integer Table Lookup, for example, `(60 * np.sin(x)).astype(np.int64)`.

### Scalars and tensors

Functions can use scalar and tensors. As with Python, it is prefered to use tensorization, to make
computations faster.

### Inputs

Inputs of a compiled function can be either encrypted or clear. Use of clear inputs is however
quite limited. Remark that constants can appear in the program
without much constraints, they are different from clear inputs which are dynamic.

## Bit width constraints

Bit width of encrypted values has a limit. We are constantly working on increasing the bit width limit.
Exceeding this limit will trigger an error.
