# Performance

This document shows some basic things you can do to improve the performance of your circuit. 

Here are some quick tips to reduce the execution time of your circuit:

- Reduce the amount of [table lookups](../core-features/table_lookups.md) in the circuit.
- Try different implementation strategies for [complex operations](../core-features/non_linear_operations.md#comparisons).
- Utilize [rounding](../core-features/rounding.md) and [truncating](../core-features/truncating.md) if your application doesn't require precise execution.
- Use tensors as much as possible in your circuits.
- Enable dataflow parallelization, by setting `dataflow_parallelize=True` in the [configuration](../guides/configure.md).
- Tweak `p_error` configuration option until you get optimal exactness vs performance tradeoff for your application.
- Specify composition when using [modules](../compilation/composing_functions_with_modules.md#optimizing-runtimes-with-composition-policies).

You can refer to our full [Optimization Guide](../optimization/self.md) for detailed examples of how to do each of these, and more!
