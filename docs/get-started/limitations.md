# Limitations
This document outlines the current limitations of Concrete, concerning the control flow constraints, the encrypted type constraints, and the bit width constraints of encrypted values.
## Control flow constraints

Concrete doesn not support some control flow statements, including the `if` and `while` statement when the condition depends on an encrypted value. However, control flow statements with constant values are allowed, for example, `for i in range(SOME_CONSTANT)`, `if os.environ.get("SOME_FEATURE") == "ON":`.

## Type constraints

Floating-point inputs or floating-point outputs are not supported. You can have floating-point intermediate values as long as they can be converted to an integer Table Lookup, for example, `(60 * np.sin(x)).astype(np.int64)`.

## Bit width constraints

Bit width of encrypted values has a limit. We are constantly working on increasing the bit width limit. Exceeding this limit will trigger an error.
