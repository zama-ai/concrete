# Future Features

As explained in [this section](fhe_and_framework_limits.md#limits-of-this-project), the **Concrete Numpy** package
is currently in its first version, and is sometimes constrained in term of functionalities. However, the good
news is that we are going to release new versions regularly, and more functionality will be added progressively.

In this page, we briefly list what the plans for next versions of **Concrete Numpy** are:
- **better performance**: further versions will contain improved versions of the **Concrete Library**, with faster
execution; also, the **Concrete Compiler** will be improved, to have faster local execution (with multi-threading
for example) and faster production execution (with distribution over a set of machines or use of hardware accelerations)
- **more complete benchmarks**: we will have an extended benchmark, containing lots of functions that you may want to compile; then, we will measure the framework progress by tracking the number of successfully compiled functions over time. Also, this public benchmark will be a way for other competing frameworks or technologies to compare fairly with us, in terms of functionality or performance
- **serialization**: we are going to add several utils, to serialize ciphertexts or keys
