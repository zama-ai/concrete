# Numpy Support

In this section, we list the operations which are supported currently in the **Concrete Framework**. Please have a look to numpy [documentation](https://numpy.org/doc/stable/user/index.html) to know what these operations are about.

## Unary operations

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

## Binary operations

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

## Indexing

Indexing is described in [this section](../tutorial/INDEXING.md).


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

```{warning}
FIXME(Arthur): speak about `astype`
```

```{warning}
FIXME(Umut): speak about `shape`
```







