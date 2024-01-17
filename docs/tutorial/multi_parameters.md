# Multi Parameters

Integers in Concrete are encrypted and processed according to a set of cryptographic parameters. By default, multiple sets of such parameters are selected by the Concrete Optimizer. This might not be the best approach for every use case, and there is the option to use mono parameters instead.

When multi parameters are enabled, a different set of parameters are selected for each bit-width in the circuit, which results in:
- Faster execution (generally).
- Slower key generation.
- Larger keys.
- Larger memory usage during execution.

To disable it, you can use `parameter_selection_strategy=fhe.ParameterSelectionStrategy.MONO` configuration option.
