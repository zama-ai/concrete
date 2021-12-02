# Future Features

As explained in [this section](FHE_AND_FRAMEWORK_LIMITS.md#concrete-framework-limits), the **Concrete Framework**
is currently in a preliminary version, and quite constrained in term of functionalities. However, the good
news is that we are going to release new versions regularly, where a lot of functionalities will be added progressively.

In this page, we briefly list what the plans for next versions of the **Concrete Framework** are:
- **better performance**: further versions will contain improved versions of the **Concrete Library**, with faster
execution; also, the **Concrete Compiler** will be improved, to have faster local execution (with multi-threading
for example) and faster production execution (with distribution over a set of machines or use of hardware accelerations)
- **more support for torch, and support for other ML frameworks**: we will continue to extend our support for torch models, and have conversions from Keras, tensorflow
- **more complete benchmarks**: we will have an extended benchmark, containing lots of functions that one day one would want to compile; then, we will measure the framework progress by tracking the number of successfully compiled functions over time. Also, this public benchmark will be a way for other competing frameworks or technologies to compare fairly with us, in terms of functionality or performance
- **Machine Learning helpers**: our midterm direction is to provide our users a set of tools to help her turn her use case in an homomorphic equivalent. This set of tools will help her reduce the needed variable precision and/or optimize the operations required to make the fastest possible compiled model.

Also, if you are especially looking for some new feature, you can drop a message to <hello@zama.ai>.



