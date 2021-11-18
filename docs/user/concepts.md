# Concepts

% Here we should describe at high level the overall design of the compiler, format
% of the manipulated elements, the goal of each pass. It will mainly introduce concepts that will later be developed
% into specific sub-sections (like Input Format / Bindings / Dialect…)

The zamacompiler is based on [MLIR](https://mlir.llvm.org/) (which is part of the LLVM project).
It has 3 internal MLIR dialects to manage different kinds of abstractions: [FHE](), [TFHE]() and [Concrete]()

It lowers programs (usually from the FHE dialect) to binaries or libraries which can be called via the [... API](). 
Its cryptographics primitives are provided by  the [concrete library](https://github.com/zama-ai/concrete).

