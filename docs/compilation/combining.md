# Combining compiled functions

In various cases, deploying a server that contains many compatible functions is important. By compatible, we mean that the functions will be used together, with outputs of some of them being used as inputs of some other ones, without decryption in the middle. It also encompasses the use of recursive functions.

To support this feature in Concrete, we have two ways:
- using the `composable` flag in the compilation, when there is a unique function. This option is described in [this document](composition.md)
- using the Concrete modules, when there are several functions, or when there is a unique function for which we want to more precisely detail how outputs are reused as further inputs. This functionality is described in [this document](composing_functions_with_modules.md)
