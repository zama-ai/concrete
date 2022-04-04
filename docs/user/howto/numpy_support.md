# Numpy Support

In this section, we list the operations which are supported currently in **Concrete Numpy**. Please have a look to numpy [documentation](https://numpy.org/doc/stable/user/index.html) to know what these operations are about.

<!--- gen_supported_ufuncs.py: inject supported operations [BEGIN] -->
<!--- do not edit, auto generated part by `python3 gen_supported_ufuncs.py` in docker -->
List of supported functions:
- absolute
- add
- arccos
- arccosh
- arcsin
- arcsinh
- arctan
- arctan2
- arctanh
- bitwise_and
- bitwise_or
- bitwise_xor
- cbrt
- ceil
- clip
- concatenate
- copysign
- cos
- cosh
- deg2rad
- degrees
- dot
- equal
- exp
- exp2
- expm1
- fabs
- float_power
- floor
- floor_divide
- fmax
- fmin
- fmod
- gcd
- greater
- greater_equal
- heaviside
- hypot
- invert
- isfinite
- isinf
- isnan
- lcm
- ldexp
- left_shift
- less
- less_equal
- log
- log10
- log1p
- log2
- logaddexp
- logaddexp2
- logical_and
- logical_not
- logical_or
- logical_xor
- matmul
- maximum
- minimum
- multiply
- negative
- nextafter
- not_equal
- positive
- power
- rad2deg
- radians
- reciprocal
- remainder
- reshape
- right_shift
- rint
- sign
- signbit
- sin
- sinh
- spacing
- sqrt
- square
- subtract
- sum
- tan
- tanh
- true_divide
- trunc
<!--- gen_supported_ufuncs.py: inject supported operations [END] -->

# Shapes

Our encrypted tensors have shapes just like numpy arrays.
We determine the shapes of the inputs from the inputset, and we infer the shapes of the intermediate values from the function that is being compiled.

You can access the shape of a tensor by accessing the `shape` property, just like in numpy.
Here is an example:
```python
def function_to_compile(x):
    return x.reshape((x.shape[0], -1))
```

One important aspect of our library is that, scalars are tensors of shape `()`.
This is transparent to you, as a user, but it's something to keep in mind, especialy if you are accessing the `shape` property in the functions that you are compiling.
This schema is used by numpy and pytorch as well.

## Indexing

Indexing is described in [this section](../tutorial/indexing.md).

## Other machine-learning-related operators

We support (sometimes, with limits) some other operators:

- dot: one of the operators must be non-encrypted
- clip: the minimum and maximum values must be constant
- transpose
- ravel
- reshape: the shapes must be constant
- flatten
- matmul: one of the two matrices must be non-encrypted. Only 2D matrix multiplication is supported for now

## Operators which are not numpy-restricted

The framework also gives support for:

- shifts, i.e., `x op y` for `op` in `[<<, >>, ]`: if one of `x` or `y` is a constant
- boolean test operations, i.e., `x op y` for `op` in `[<, <=, ==, !=, >, >=]`: if one of `x` or `y` is a constant
- boolean operators, i.e., `x op y` for `op` in `[&, ^, |]`: if one of `x` or `y` is a constant
- powers, i.e., `x ** y`: if one of `x` or `y` is a constant
- modulo, i.e., `x % y`: if one of `x` or `y` is a constant
- invert, i.e., `~x`
- true div, i.e., `x / y`: if one of `x` or `y` is a constant
- floor div, i.e., `x // y`: if one of `x` or `y` is a constant

There is support for astype as well, e.g. `x.astype(numpy.int32)`. This allows to control which data type to use for computations. In the context of FHE going back to integers may allow to fuse floating point operations together, see [this tutorial](../tutorial/working_with_floating_points.md) to see how to work with floating point values.
