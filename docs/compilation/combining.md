# Combining compiled functions

This document explains how to combine compiled functions in **Concrete**, focusing on scenarios where multiple functions need to work together seamlessly. The goal is to ensure that outputs from certain functions can be used as inputs for others without decryption, including in recursive functions.

**Concrete** offers two methods to achieve this:

- **Using the `composable` flag**: This method is suitable when there is a single function. The composable flag allows the function to be compiled in a way that its output can be used as input for subsequent operations. For more details, refer to the [composition documentation](composition.md).

- **Using Concrete modules**: This method is ideal when dealing with multiple functions or when more control is needed over how outputs are reused as inputs. Concrete modules allow you to specify precisely how functions interact. For further information, see the [modules documentation](composing_functions_with_modules.md).

