# Multi Parameters

Integers in Concrete are encrypted and processed according to a set of cryptographic parameters. By default, only a single set of such parameters are selected by Concrete Optimizer. This is not the best approach for every use case so multi parameters are introduced.

When they are enabled, a different set of parameters are selected for each bit-width in the circuit, which results in:
- Faster execution (generally).
- Slower key generation.
- Larger keys.
- Larger memory usage during execution.

To enable them, you can use `parameter_selection_strategy=fhe.ParameterSelectionStrategy.MULTI` configuration option.
