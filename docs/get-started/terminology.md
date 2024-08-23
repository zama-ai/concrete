# Terminology and Structure

This document provides clear definitions of key concepts used in **Concrete** framework.

##### Computation graph
A data structure to represent a computation. It takes the form of a directed acyclic graph where nodes represent inputs, constants, or operations.

##### Tracing
A method that takes a Python function provided by the user and generates a corresponding computation graph.

##### Bounds
The minimum and the maximum value that each node in the computation graph can take. Bounds are used to determine the appropriate data type (for example, uint3 or int5) for each node before the computation graphs are converted to MLIR. **Concrete** simulates the graph with the inputs in the inputset to record the minimum and the maximum value for each node.

##### Circuit
The result of compilation. A circuit includes both client and server components. It has methods for various operations, such as printing and evaluation.

##### Table Lookup (TLU)
TLU stands for instructions in the form of y = T[i]. In FHE, this operation is performed with Programmable Bootstrapping, which is the equivalent operation on encrypted values. To learn more about TLU, refer to the [Table Lookup basic](../core-features/table_lookups.md) and tge [Table Lookup advanced](../core-features/table_lookups_advanced.md) section.

##### Programmable Bootstrapping (PBS)
PBS is equivalent to table lookup `y = T[i]` on encrypted values, which means that the inputs `i` and the outputs `y` are encrypted, but the table `T` is not encrypted. You can find a more detailed explanation in the [FHE Overview](../core-features/fhe_basics.md#noise-and-bootstrap).

##### TFHE
TFHE is a Fully Homomorphic Encryption (FHE) scheme that allows you to perform computations over encrypted data. For in-depth explanation of the TFHE scheme, read our blog post series [TFHE Deep Dive](https://www.zama.ai/post/tfhe-deep-dive-part-1).

