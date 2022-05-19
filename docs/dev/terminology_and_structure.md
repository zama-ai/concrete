# Terminology and Structure

## Terminology

In this section we will go over some terms that we use throughout the project.

- intermediate representation
    - a data structure to represent a computation
    - basically a computation graph in which nodes are either inputs, constants, or operations on other nodes
- tracing
    - it is the technique to take a python function from a user and generate intermediate representation corresponding to it in a painless way for the user
- bounds
    - before intermediate representation is converted to MLIR, we need to know which node will output which type (e.g., uint3 vs uint5)
    - there are several ways to do this but the simplest one is to evaluate the intermediate representation with some combinations of inputs and remember the maximum and the minimum values for each node, which is what we call bounds, and bounds can be used to determine the appropriate type for each node
- circuit
   - it is the result of compilation
   - it is made of the computation graph and the compiler engine
   - it has methods for printing, visualizing, and evaluating

## Module structure

In this section, we will discuss the module structure of **concrete-numpy** briefly. You are encouraged to check individual `.py` files to learn more!

- concrete
    - numpy
        - dtypes: data type specifications
        - values: value specifications (i.e., data type + shape + encryption status)
        - representation: representation of computation
        - tracing: tracing of python functions
        - extensions: custom functionality which is not available in numpy (e.g., conv2d)
        - mlir: mlir conversion
        - compilation: compilation from python functions to circuits
