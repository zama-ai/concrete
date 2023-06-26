# Optimizing Crypto Parameters

`concrete-optimizer` is a tool that selects appropriate technical quantities for a given fully homomorphic encryption (FHE) calculation. These quantities have an impact on the security, correctness, and efficiency of the calculation.

The security of the calculation is guaranteed to a required level, typically 128 bits. The correctness of the calculation is guaranteed up to a given error rate. The efficiency is optimized to reduce the time to execute the FHE calculus.

These quantities, known as crypto-parameters, only exist due to the FHE nature of the calculus. The crypto-parameters are in a finite but huge space. The role of the optimizer is to quickly find the most efficient crypto-parameters possible while guaranteeing security and correctness.

## Security, Correctness, and Efficiency

### Security

The security level is chosen by the user.

We typically operate at a fixed security level, such as 128 bits, to ensure that there is never a trade-off between security and efficiency. This constraint imposes a minimum amount of noise in any calculation and in public material.

An independent public research tool, the lattice estimator, estimates the security level.

The lattice estimator is a joint work from FHE experts. It estimates the security level of the couple noise and the mask size of the ciphertext (i.e. one of the crypto-parameters).

For each security level, a parameter curve of the appropriate minimal error level is pre-computed using the lattice estimator and used by the optimizer. Learn more about the parameter curve [here](https://www.notion.so/tools/parameter-curves/README.rst).

### Correctness

Correctness decreases as the level of noise increases.

Noise accumulates during calculations until it is actively reduced. Too much noise can render the calculation's result inaccurate or, worse, completely incorrect.

Before optimization, we compute a noise bound that guarantees a given error level if the noise always stays under that bound before we reduce it or before the result is decrypted. The noise growth depends on a critical quantity: the norm2 of any dot product (or equivalent) present in the calculus. This norm2 changes the scale of the noise, so we must reduce it sufficiently for the next dot product operation whenever we reduce the noise.

The user can choose the degree of correctness using two options: the PBS error probability and the global error probability.

The PBS error probability controls correctness locally, while the global error probability focuses on the overall calculation result. In the end, they both control the same elements, but one option may be easier to use depending on the specific use case.

### Efficiency

Efficiency decreases as more noise must be reduced.
Efficiency decreases as more precision are required, e.g. 7bits versus 8bits.

The cost of noise reduction increases as the scale of the noise reduction increases. The larger the norm2, the more the noise will grow after noise reduction, and the greater the cost will be to reduce it sufficiently to stay within the noise bound.

## How are the parameters optimized?

First, all optimization schemes prioritize security and correctness as hard constraints, meaning that the crypto-parameters are always slightly better than what the user queried.

In the simplest case, the optimizer performs an exhaustive search in the full parameter space and selects the best solution. While the space to explore is huge, exact lower bound cuts are used to avoid exploring guaranteed non-interesting regions. This makes the process both fast and exhaustive. The simplest case is called mono-parameter, where all parameters are shared between all calculus operations.

In more complex cases, the optimizer iteratively performs an exhaustive search with lower bound cuts in a wide subspace of the full parameter space until it converges to a locally optimal solution. Since the wide subspace is large and highly multi-dimensional, it cannot be trapped in a poor locally optimal solution. The more complex case is called multi-parameters, where different calculus operations have tailored parameters.

## How can I figure understand and explore crypto-parameters

One can have a look at [reference crypto-parameters](../../compilers/concrete-optimizer/v0-parameters/ref/v0_last_128) for each security (but for a given correctness).
This provides insight between the calcululs content (i.e. maximum precision, maximum dot norm2) and the cost.

Then one can manually explore crypto-parameters space using a [CLI tool](../../compilers/concrete-optimizer/v0-parameters/README.md).
