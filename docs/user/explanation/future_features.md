# Future Features

As explained in [this section](fhe_and_framework_limits.md#limits-of-this-project), the **Concrete Numpy** package
is currently in its first version, and quite constrained in term of functionalities. However, the good
news is that we are going to release new versions regularly, where a lot of functionalities will be added progressively.

In this page, we briefly list what the plans for next versions of **Concrete Numpy** are:
- **better performance**: further versions will contain improved versions of the **Concrete Library**, with faster
execution; also, the **Concrete Compiler** will be improved, to have faster local execution (with multi-threading
for example) and faster production execution (with distribution over a set of machines or use of hardware accelerations)
- **more complete benchmarks**: we will have an extended benchmark, containing lots of functions that one day one would want to compile; then, we will measure the framework progress by tracking the number of successfully compiled functions over time. Also, this public benchmark will be a way for other competing frameworks or technologies to compare fairly with us, in terms of functionality or performance
- **client/server APIs**: today, the `run` function is performing the key generation, the encryption, the inference and the decryption to allow machine learning practitioners to test both performance and accuracy of FHE friendly models. Soon, we are going to have separated APIs to perform the steps one by one, and thus, a full client / server API
- **serialization**: we are going to add several utils, to serialize ciphertexts or keys

## Regarding machine learning

We will continue to consider our `NPFHECompiler` class (compilation of numpy programs) as the main entry point for **Concrete Numpy**. In the future, we may move all ML tools currently present in **Concrete Numpy** to a new to-be-named ML specific package.

Our plans to extend machine learning support in the future are:

- **extend support for torch**: having more layers and more complex `forward `patterns, and also having ready to use neural networks and neural network blocks that are compatible with FHE
- **support for other ML frameworks**: we will provide FHE compatible model architectures for classical ML models which will be trainable with popular frameworks such as sklearn. Tools for quantization aware training and FHE compatible algorithms are also in our plans

Also, if you are especially looking for some new feature, you can drop a message to <hello@zama.ai>.



