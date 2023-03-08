# Terminology and Structure

## Terminology

Some terms used throughout the project include:

* computation graph - a data structure to represent a computation. This is basically a directed acyclic graph in which nodes are either inputs, constants or operations on other nodes.
* tracing - the technique that takes a Python function from the user and generates the corresponding computation graph in an easy-to-read format.
* bounds - before a computation graph is converted to MLIR, we need to know which node will output which type (e.g., uint3 vs euint5). Computation graphs with different inputs must remember the minimum and maximum values for each node, which is what we call bounds, and use bounds to determine the appropriate type for each node.
* circuit - the result of compilation. A circuit is made of the client and server components and has methods, everything from printing to evaluation.

## Module structure

In this section, we will briefly discuss the module structure of **Concrete-Numpy**. You are encouraged to check individual `.py` files to learn more.

* Concrete
  * Numpy
    * dtypes - data type specifications
    * values - value specifications (i.e., data type + shape + encryption status)
    * representation - representation of computation
    * tracing - tracing of Python functions
    * extensions - custom functionality which is not available in NumPy (e.g., direct table lookups)
    * MLIR - MLIR conversion
    * compilation - compilation from a Python function to a circuit, client/server architecture
  * ONNX
    * convolution - custom convolution operations that follow the behavior of ONNX
