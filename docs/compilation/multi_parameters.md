# Multi parameters
This document explains the implications and configuration of multi parameters in **Concrete**.

In **Concrete**, integers are encrypted and processed based on a set of cryptographic parameters. By default, the **Concrete** optimizer selects multiple sets of these parameters, which may not be optimal for every use case. In such cases, you can choose to use mono parameters instead.

When multi parameters are enabled, the optimizer selects a different set of parameters for each bit-width in the circuit. This approach has several implications:

* Faster execution in general
* Slower key generation
* Larger keys
* Larger memory usage during execution

When enabled, you can control the level of circuit partitioning by setting the **multi\_parameter\_strategy** as described in [configuration](../guides/configure.md#options).

To disable multi parameters, use `parameter_selection_strategy=fhe.ParameterSelectionStrategy.MONO` configuration option.


