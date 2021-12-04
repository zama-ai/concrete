# Benchmarks

```{warning}
FIXME(Umut): update a bit
```

In order to track our progress over time, we have set a [public benchmark](https://progress.zama.ai) containing:
- a list of functions that we want to compile
- status on the compilation of these functions
- compilation time
- evaluation time on different hardware's
- accuracy of the functions for which it makes sense
- loss of the functions for which it makes sense

Remark that we are not limited to these, and we'll certainly add more information later, as key generation time, encryption and decryption time, and more evaluation time once the explicit inference API is available.

The benchmark can be used by competitive frameworks or technologies, in order to compare fairly with the **Concrete Framework**. Notably, one can see:
- if the same functions can be compiled
- what are discrepancies in the exactness of the evaluations
- how do evaluation times compare

If one wants to see more functions in the benchmark or if there is more information you would like the benchmark to track, don't hesitate to drop an email to <hello@zama.ai>.