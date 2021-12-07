# Working With Floating Points

## An example

```python
import numpy as np
import concrete.numpy as hnp

# Function using floating points values converted back to integers at the end
def f(x):
    return np.fabs(50 * (2 * np.sin(x) * np.cos(x))).astype(np.uint32)
    # astype is to go back to the integer world

# Compiling with x encrypted
compiler = hnp.NPFHECompiler(f, {"x": "encrypted"})
compiler.eval_on_inputset(range(64))

circuit = compiler.get_compiled_fhe_circuit()

assert circuit.run(3) == f(3)
assert circuit.run(0) == f(0)
assert circuit.run(1) == f(1)
assert circuit.run(10) == f(10)
assert circuit.run(60) == f(60)

print("All good!")
```

## Supported operations

The following operations are supported in the latest release, and we'll add more operations in the upcoming releases.

<!--- gen_supported_ufuncs.py: inject supported operations [BEGIN] -->
<!--- do not edit, auto generated part by `python3 gen_supported_ufuncs.py` in docker -->
List of supported unary functions:
- absolute
- arccos
- arccosh
- arcsin
- arcsinh
- arctan
- arctanh
- cbrt
- ceil
- cos
- cosh
- deg2rad
- degrees
- exp
- exp2
- expm1
- fabs
- floor
- isfinite
- isinf
- isnan
- log
- log10
- log1p
- log2
- logical_not
- negative
- positive
- rad2deg
- radians
- reciprocal
- rint
- sign
- signbit
- sin
- sinh
- spacing
- sqrt
- square
- tan
- tanh
- trunc

List of supported binary functions if one of the two operators is a constant scalar:
- arctan2
- bitwise_and
- bitwise_or
- bitwise_xor
- copysign
- equal
- float_power
- floor_divide
- fmax
- fmin
- fmod
- gcd
- greater
- greater_equal
- heaviside
- hypot
- lcm
- ldexp
- left_shift
- less
- less_equal
- logaddexp
- logaddexp2
- logical_and
- logical_or
- logical_xor
- maximum
- minimum
- nextafter
- not_equal
- power
- remainder
- right_shift
- true_divide
<!--- gen_supported_ufuncs.py: inject supported operations [END] -->

```{warning}
FIXME(Benoit): see what kind of other supported operations we could list here
```

## Limitations

Floating point support in **Concrete** is very limited for the time being. They can't appear on inputs, or they can't be outputs. However, they can be used in intermediate results. Unfortunately, there are limitations on that front as well.

This biggest one is that, because floating point operations are fused into table lookups with a single unsigned integer input and single unsigned integer output, only univariate portion of code can be replaced with table lookups, which means multivariate portions cannot be compiled.

To give a precise example, `100 - np.fabs(50 * (np.sin(x) + np.sin(y)))` cannot be compiled because the floating point part depends on both `x` and `y` (i.e., it cannot be rewritten in the form `100 - table[z]` for a `z` that could be computed easily from `x` and `y`).

To dive into implementation details, you may refer to [Fusing Floating Point Operations](../../dev/explanation/FLOAT-FUSING.md) document.
