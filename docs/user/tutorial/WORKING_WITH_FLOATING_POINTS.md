# Working With Floating Points

## An example

```python
def f(x):
    np.fabs(100 * (2 * np.sin(x) * np.cos(x))).astype(np.uint32) # astype is to go back to integer world
```

where

- `x = EncryptedScalar(UnsignedInteger(bits))`

results in

```python
engine.run(3) == 27
engine.run(0) == 0
engine.run(1) == 90
engine.run(10) == 91
engine.run(60) == 58
```

## Supported operations

The following operations are supported in the latest release, and we'll add more operations in the upcoming releases.

- np.arccos
- np.arccosh
- np.arcsin
- np.arcsinh
- np.arctan
- np.arctanh
- np.cbrt
- np.ceil
- np.cos
- np.cosh
- np.deg2rad
- np.degrees
- np.exp
- np.exp2
- np.expm1
- np.fabs
- np.floor
- np.log
- np.log10
- np.log1p
- np.log2
- np.rad2deg
- np.radians
- np.rint
- np.sin
- np.sinh
- np.spacing
- np.sqrt
- np.tan
- np.tanh
- np.trunc

## Limitations

Floating point support in **concrete** is very limited for the time being. They can't appear on inputs, or they can't be outputs. However, they can be used in intermediate results. Unfortunately, there are limitations on that front as well.

This biggest one is that, because floating point operations are fused into table lookups with a single unsigned integer input and single unsigned integer output, only univariate portion of code can be replaced with table lookups, which means multivariate portions cannot be compiled.

To give a precise example, `100 - np.fabs(50 * (np.sin(x) + np.sin(y)))` cannot be compiled because the floating point part depends on both `x` and `y` (i.e., it cannot be rewritten in the form `100 - table[z]` for a `z` that could be computed easily from `x` and `y`).

To dive into implementation details, you may refer to [Fusing Floating Point Operations](../../dev/explanation/FLOAT-FUSING.md) document.
