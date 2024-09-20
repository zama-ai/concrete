# Examples

This document gives an overview of the structure of the examples, which are tutorials containing more or less elaborated usages of Concrete, to showcase its functionality on practical use cases. Examples are either provided as a Python script or a Jupyter notebook.

## Concrete Python

### How to create an example?

#### Jupyter notebook example

- Create `examples/foo/foo.ipynb`
- Write the example in the notebook
- The notebook will be executed in the CI with `make test-notebooks` target

#### Python script example

- Create `examples/foo/foo.py`
- Write the example in the script
  - Example should contain a class called `Foo`
  - `Foo` should have the following arguments in its `__init__`:
    - configuration: Optional\[fhe.Configuration] = None
    - compiled: bool = True
  - It should compile the circuit with an appropriate inputset using the given configuration if compiled is true
  - It should have any additional common utilities (e.g., encoding/decoding) shared between the tests and the benchmarks
- Then, add tests for the implementation in `tests/execution/test_examples.py`
- Optionally, create `benchmarks/foo.py` and [add benchmarks](benchmarking.md#how-to-add-new-benchmarks).
