# Optimizing Crypto Parameters

`concrete-optimizer` is a tool that selects appropriate appropriate cryptographic parameters for a given fully homomorphic encryption (FHE) computation. These quantities have an impact on the security, correctness, and efficiency of the computation.

The computation is guaranteed to be secure with the given level of security (see [here](../security/security_curves.md) for details) typically 128 bits. The correctness of the computation is guaranteed up to a given failure probability. A surrogate of the execution time is minimized which allow efficient FHE computation.

The cryptographic parameters are degrees of freedom in the FHE algorithms (bootstrapping, keyswitch, ...) that need to be fixed. The crypto-parameters are in a finite but huge space. The role of the optimizer is to quickly find the most efficient crypto-parameters possible while guaranteeing security and correctness.

## Security, Correctness, and Efficiency

### Security

The security level is chosen by the user.

We typically operate at a fixed security level, such as 128 bits, to ensure that there is never a trade-off between security and efficiency. This constraint imposes a minimum amount of noise in any ciphertext.

An independent public research tool, the lattice estimator, estimates the security level.

The lattice estimator is a joint work from FHE experts. It estimates the security level of the couple noise and the secret key size of the ciphertext (i.e. one of the crypto-parameters).

For each security level, a parameter curve of the appropriate minimal error level is pre-computed using the lattice estimator and used by the optimizer. Learn more about the parameter curve [here](../security/security_curves.md).

### Correctness

Correctness decreases as the level of noise increases.

Noise accumulates during computations until it is actively reduced. Too much noise can render the computation's result inaccurate or, worse, completely incorrect.

Before optimization, we compute a noise bound that guarantees a given error level if the noise always stays under that bound before we reduce it or before the result is decrypted. The noise growth depends on a critical quantity: the 2-norm of any dot product (or equivalent) present in the calculus. This 2-norm changes the scale of the noise, so we must reduce it sufficiently for the next dot product operation whenever we reduce the noise.

The user can choose the degree of correctness using two options: the PBS error probability and the global error probability.

The PBS error probability controls correctness locally, while the global error probability focuses on the overall computation result. In the end, they both control the same elements, but one option may be easier to use depending on the specific use case.

### Efficiency

Efficiency decreases as more precision are required, e.g. 7bits versus 8bits.

The larger the 2-norm is, the bigger noise will be after a dot product. To still remain below the noise bound, the inputs of the dot product must have a noise sufficiently small. The smaller this noise is, the slower the previous bootstrapping are. Therefore, the larger the 2norm is, the slower the computation will be.

## How are the parameters optimized?

First, the optimization prioritizes security and correctness meaning that the security (or the probability of correctness) could be in practice a bit higher than the requested one.

In the simplest case, the optimizer performs an exhaustive search in the full parameter space and selects the best solution. While the space to explore is huge, exact lower bound cuts are used to avoid exploring guaranteed non-interesting regions. This makes the process both fast and exhaustive. The simplest case is called mono-parameter, where all parameters are shared by the whole computation graph.

In more complex cases, the optimizer iteratively performs an exhaustive search with lower bound cuts in a wide subspace of the full parameter space until it converges to a locally optimal solution. Since the wide subspace is large and highly multi-dimensional, it should not be trapped in a poor locally optimal solution. The more complex case is called multi-parameters, where different calculus operations have tailored parameters.

## How can I figure understand and explore crypto-parameters

One can have a look at [reference crypto-parameters](../../../compilers/concrete-optimizer/v0-parameters/ref/v0_last_128) for each security (but for a given correctness).
This provides insight between the calcululs content (i.e. maximum precision, maximum dot 2-norm) and the cost.

Then one can manually explore crypto-parameters space using a [CLI tool](../../../compilers/concrete-optimizer/v0-parameters/README.md).

## Citing

If you use this tool in your work, please cite:
> Bergerat, Loris and Boudi, Anas and Bourgerie, Quentin and Chillotti, Ilaria and Ligier, Damien and Orfila Jean-Baptiste and Tap, Samuel, Parameter Optimization and Larger Precision for (T)FHE, Journal of Cryptology, 2023, Volume 36

A pre-print is available as

Cryptology ePrint Archive, [Paper 2022/704](https://eprint.iacr.org/2022/704)
