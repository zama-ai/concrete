# Optimizing Crypto Parameters

`concrete-optimizer` is a tool that selects appropriate cryptographic parameters for a given fully homomorphic encryption (FHE) computation. These parameters have an impact on the security, correctness, and efficiency of the computation.

The computation is guaranteed to be secure with the given level of security (see [here](./security.md) for details) which is typically 128 bits. The correctness of the computation is guaranteed up to a given failure probability. A surrogate of the execution time is minimized which allows for  efficient FHE computation.

The cryptographic parameters are degrees of freedom in the FHE algorithms (bootstrapping, keyswitching, etc.) that need to be fixed. The search space for possible crypto-parameters is finite but extremely large. The role of the optimizer is to quickly find the most efficient crypto-parameters possible while guaranteeing security and correctness.

## Security, Correctness, and Efficiency

### Security

The security level is chosen by the user. We typically operate at a fixed security level, such as 128 bits, to ensure that there is never a trade-off between security and efficiency. This constraint imposes a minimum amount of noise in all ciphertexts.

An independent public research tool, the [lattice estimator](https://github.com/malb/lattice-estimator), is used to estimate the security level. The lattice estimator is maintained by FHE experts. For a given set of crypto-parameters, this tool considers all possible attacks and returns a security level.

For each security level, a parameter curve of the appropriate minimal error level is pre-computed using the lattice estimator, and is used as an input to the optimizer. Learn more about the parameter curves [here](./security.md).

### Correctness

Correctness decreases as the level of noise increases. Noise accumulates during homomorphic computation until it is actively reduced via bootstrapping. Too much noise can lead to the result of a computation being inaccurate or completely incorrect.

Before optimization, we compute a noise bound that guarantees a given error level (under the assumption that noise growth is correctly managed via bootstrapping). The noise growth depends on a critical quantity: the 2-norm of any dot product (or equivalent) present in the calculus. This 2-norm changes the scale of the noise, so we must reduce it sufficiently for the next dot product operation whenever we reduce the noise.

The user can control error probability in two ways: via the PBS error probability and the global error probability.

The PBS error probability controls correctness locally (i.e., represents the error probability of a _single_ PBS operation), while the global error probability focuses on the overall computation result (i.e., represents the error probability of the _entire_ computation). These probabilities are related, and choosing which one to use may depend on the specific use case.

### Efficiency

Efficiency decreases as more precision is required, e.g. 7-bits versus 8-bits. The larger the 2-norm is, the bigger the noise will be after a dot product. To remain below the noise bound, we must ensure that the inputs to the dot product have a sufficiently small noise level. The smaller this noise is, the slower the previous bootstrapping will be. Therefore, the larger the 2norm is, the slower the computation will be.

## How are the parameters optimized

The optimization prioritizes security and correctness. This means that the security level (or the probability of correctness) could, in practice, be a bit higher than the level which is requested by the user.

In the simplest case, the optimizer performs an exhaustive search in the full parameter space and selects the best solution. While the space to explore is huge, exact lower bound cuts are used to avoid exploring regions which are guaranteed to not contain an optimal point. This makes the process both fast and exhaustive. This case is called mono-parameter, where all parameters are shared by the whole computation graph.

In more complex cases, the optimizer iteratively performs an exhaustive search, with lower bound cuts in a wide subspace of the full parameter space, until it converges to a locally optimal solution. Since the wide subspace is large and multi-dimensional, it should not be trapped in a poor locally optimal solution. The more complex case is called multi-parameter, where different calculus operations have tailored parameters.

## How can I determine, understand, and explore crypto-parameters

One can have a look at [reference crypto-parameters](../../compilers/concrete-optimizer/v0-parameters/ref/v0_last_128) for each security level (but for a given correctness). This provides insight between the calcululs content (i.e. maximum precision, maximum dot 2-norm, etc.,) and the cost.

Then one can manually explore crypto-parameters space using a [CLI tool](../../compilers/concrete-optimizer/v0-parameters/README.md).

## Citing

If you use this tool in your work, please cite:
> Bergerat, Loris and Boudi, Anas and Bourgerie, Quentin and Chillotti, Ilaria and Ligier, Damien and Orfila Jean-Baptiste and Tap, Samuel, Parameter Optimization and Larger Precision for (T)FHE, Journal of Cryptology, 2023, Volume 36

A pre-print is available as Cryptology ePrint Archive [Paper 2022/704](https://eprint.iacr.org/2022/704)
