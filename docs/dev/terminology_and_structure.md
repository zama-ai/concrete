# Terminology and Structure

## Terminology

Some terms used throughout the project include:

* computation graph: a data structure to represent a computation. This is basically a directed acyclic graph in which nodes are either inputs, constants or operations on other nodes.
* tracing: a technique that takes a python function from the user and generates a corresponding computation graph
* bounds: before computation graphs are converted to MLIR, we need to know which value should have which type (e.g., uint3 vs int5). we use inputsets to do that. we simulate the graph with the inputs in the inputset to remember the minimum and the maximum value for each node, which is what we call bounds, and use bounds to determine the appropriate type for each node.
* circuit: the result of compilation. a circuit is made of the client and server components. it has methods for everything from printing to evaluation.

## Module structure

In this section, we will briefly discuss the module structure of **Concrete-Python**. You are encouraged to check individual `.py` files to learn more.

* concrete
  * fhe
    * dtypes: data type specifications (e.g., int4, uint5, float32)
    * values: value specifications (i.e., data type + shape + encryption status)
    * representation: representation of computation (e.g., computation graphs, nodes)
    * tracing: tracing of python functions
    * extensions: custom functionality (see [Extensions](../tutorial/extensions.md))
    * mlir: computation graph to mlir conversion
    * compilation: configuration, compiler, artifacts, circuit, client/server, and anything else related to compilation
