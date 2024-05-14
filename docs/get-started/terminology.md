# Terminology and Structure

This document provides clear definitions of key concepts used in **Concrete** framework.

* **Computation graph:** A data structure to represent a computation. It takes the form of a directed acyclic graph where nodes represent inputs, constants, or operations.

* **Tracing:** A method that takes a Python function provided by the user and generates a corresponding computation graph.

* **Bounds:** The minimum and the maximum value that each node in the computation graph can take. Bounds are used to determine the appropriate data type (for example, uint3 or int5) for each node before the computation graphs are converted to MLIR. **Concrete** simulates the graph with the inputs in the inputset to record the minimum and the maximum value for each node.

* **Circuit:** The result of compilation. A circuit includes both client and server components. It has methods for various operations, such as printing and evaluation.

