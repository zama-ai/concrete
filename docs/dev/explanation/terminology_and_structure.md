# Terminology and Structure

## Terminology

In this section we will go over some terms that we use throughout the project.

- intermediate representation
    - a data structure to represent a calculation
    - basically a computation graph where nodes are either inputs or operations on other nodes
- tracing
    - it is our technique to directly take a plain numpy function from a user and deduce its intermediate representation in a painless way for the user
- bounds
    - before intermediate representation is sent to the compiler, we need to know which node will output which type (e.g., uint3 vs uint5)
    - there are several ways to do this but the simplest one is to evaluate the intermediate representation with all combinations of inputs and remember the maximum and the minimum values for each node, which is what we call bounds, and bounds can be used to determine the appropriate type for each node
- fhe circuit
   - it is the result of compilation
   - it contains the operation graph and the compiler engine in it
   - it has methods for printing, visualizing, and evaluating

## Module structure

In this section, we will discuss the module structure of **concrete-numpy** briefly. You are encouraged to check individual `.py` files to learn more!

- concrete
    - common: types and utilities that can be used by multiple frontends (e.g., numpy, torch)
      - bounds_measurement: utilities for determining bounds of intermediate representation
      - common_helpers: various utilities
      - compilation: type definitions related to compilation (e.g., compilation config, compilation artifacts)
      - data_types: type definitions of typing information of intermediate representation
      - debugging: utilities for printing/displaying intermediate representation
      - extensions: utilities that provide special functionality to our users
      - fhe_circuit: class to hold the result of the compilation
      - helpers: various helpers
      - mlir: MLIR conversion module
      - operator_graph: code to wrap and make manipulating networkx graphs easier
      - optimization: optimization and simplification
      - representation: type definitions of intermediate representation
      - tracing: utilities for generic function tracing used during intermediate representation creation
      - values: define the different types we use, including tensors and scalar, encrypted or clear
    - numpy: numpy frontend of the package
        - compile: compilation of a numpy function
        - np_dtypes_helpers: utilities about types
        - np_fhe_compiler: main API for compilation of numpy functions
        - np_indexing_helpers: utilities for indexing
        - np_inputset_helpers: utilities for inputsets
        - np_mlir_converter: utilities for MLIR conversion
        - tracing: tracing of numpy functions
