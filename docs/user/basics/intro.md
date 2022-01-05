
# What is **Concrete Numpy**?

## Introduction

**Concrete Numpy**, or **Concrete** for short, is an open-source set of tools which aims to simplify the use of so-called fully homomorphic encryption (FHE) for data scientists.

FHE is a powerful cryptographic tool, which allows servers to perform computations directly on encrypted data without needing to decrypt first. With FHE, privacy is at the center, and you can build services which ensure full privacy of the user and are the perfect equivalent of their unsecure counterpart.

FHE is also a killer feature regarding data breaches: as anything done on the server is done over encrypted data, even if the server is compromised, there is in the end no leak of useful data.

With **Concrete Numpy**, data scientists can implement machine learning models using a [subset of numpy](../howto/numpy_support.md) that compile to FHE. They will be able to train models with popular machine learning libraries and then convert the prediction functions of these models, that they write in numpy, to FHE.

**Concrete Numpy** is made of several parts:
- an entry API, which is the main function of the so-called **Concrete frontend**, which takes programs made from a subset of numpy, and converts them to an FHE program
- the **Concrete compiler**, which is called by the frontend, which allows you to turn an MLIR program into an FHE program, on the top of **Concrete Library**, which contains the core cryptographic APIs for computing with FHE;
- some ML tools, in an early version, allowing for example to turn some torch programs into numpy, and then to use the main API stack to finally get an FHE program.

In a further release, **Concrete Numpy** will be divided into a **Concrete Framework** package, containing the compiler, the core lib and the frontend(s), and in a **Concrete ML**, which will contain ML tools, made on top of the **Concrete Framework**. Names of these packages are succeptible to change.

## Organization of the documentation

Basically, we have divided our documentation into several parts:
- one about basic elements, notably a description of the installation, that you are currently reading
- one dedicated to _users_ of **Concrete Numpy**, with tutorials, how-tos and deeper explanations
- one detailing the APIs of the different functions of the frontend, directly done by parsing its source code
- and finally, one dedicated to _developers_ of **Concrete Numpy**, who could be internal or external contributors to the framework

## A work in progress

```{note}
Concrete is a work in progress, and is currently limited to a certain number of operators and features. In the future, there will be improvements as described in this [section](../explanation/future_features.md).
```

The main _current_ limits are:
- **Concrete** only supports unsigned integers
- **Concrete** needs integers to fit in a maximum of 7 bits
- **Concrete** computations are exact (except a very small probability) for computations on 6 bits or less, and exact at a probability close to 90% for 7 bits computations

To overcome the above limitations, Concrete has a [popular quantization](../explanation/quantization.md) method built in the framework that allows map floating point values to integers. We can [use this approach](../howto/use_quantization.md) to run models in FHE. Lastly, we give hints to the user on how to [reduce the precision](../howto/reduce_needed_precision.md) of a model to make it work in Concrete.
