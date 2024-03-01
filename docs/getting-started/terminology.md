# Terminology and Structure

## Terminology

Some terms used throughout the project include:

* **computation graph:** A data structure to represent a computation. This is basically a directed acyclic graph in which nodes are either inputs, constants, or operations on other nodes.
* **tracing:** A technique that takes a Python function from the user and generates a corresponding computation graph.
* **bounds:** Before computation graphs are converted to MLIR, we need to know which value should have which type (e.g., uint3 vs int5). We use inputsets for this purpose. We simulate the graph with the inputs in the inputset to remember the minimum and the maximum value for each node, which is what we call bounds, and use bounds to determine the appropriate type for each node.
* **circuit:** The result of compilation. A circuit is made of the client and server components. It has methods for everything from printing to evaluation.

