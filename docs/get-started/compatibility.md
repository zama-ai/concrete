# Compatibility

## Supported operations

Here are the operations you can use inside the function you are compiling:

{% hint style="info" %}
Some of these operations are not supported between two encrypted values. A detailed error will be raised if you try to do something that is not supported.
{% endhint %}

### Supported Python operators.

* [\_\_abs\_\_](https://docs.python.org/3/reference/datamodel.html#object.\_\_abs\_\_)
* [\_\_add\_\_](https://docs.python.org/3/reference/datamodel.html#object.\_\_add\_\_)
* [\_\_and\_\_](https://docs.python.org/3/reference/datamodel.html#object.\_\_and\_\_)
* [\_\_eq\_\_](https://docs.python.org/3/reference/datamodel.html#object.\_\_eq\_\_)
* [\_\_floordiv\_\_](https://docs.python.org/3/reference/datamodel.html#object.\_\_floordiv\_\_)
* [\_\_ge\_\_](https://docs.python.org/3/reference/datamodel.html#object.\_\_ge\_\_)
* [\_\_getitem\_\_](https://docs.python.org/3/reference/datamodel.html#object.\_\_getitem\_\_)
* [\_\_gt\_\_](https://docs.python.org/3/reference/datamodel.html#object.\_\_gt\_\_)
* [\_\_invert\_\_](https://docs.python.org/3/reference/datamodel.html#object.\_\_invert\_\_)
* [\_\_le\_\_](https://docs.python.org/3/reference/datamodel.html#object.\_\_le\_\_)
* [\_\_lshift\_\_](https://docs.python.org/3/reference/datamodel.html#object.\_\_lshift\_\_)
* [\_\_lt\_\_](https://docs.python.org/3/reference/datamodel.html#object.\_\_lt\_\_)
* [\_\_matmul\_\_](https://docs.python.org/3/reference/datamodel.html#object.\_\_matmul\_\_)
* [\_\_mod\_\_](https://docs.python.org/3/reference/datamodel.html#object.\_\_mod\_\_)
* [\_\_mul\_\_](https://docs.python.org/3/reference/datamodel.html#object.\_\_mul\_\_)
* [\_\_ne\_\_](https://docs.python.org/3/reference/datamodel.html#object.\_\_ne\_\_)
* [\_\_neg\_\_](https://docs.python.org/3/reference/datamodel.html#object.\_\_neg\_\_)
* [\_\_or\_\_](https://docs.python.org/3/reference/datamodel.html#object.\_\_or\_\_)
* [\_\_pos\_\_](https://docs.python.org/3/reference/datamodel.html#object.\_\_pos\_\_)
* [\_\_pow\_\_](https://docs.python.org/3/reference/datamodel.html#object.\_\_pow\_\_)
* [\_\_radd\_\_](https://docs.python.org/3/reference/datamodel.html#object.\_\_radd\_\_)
* [\_\_rand\_\_](https://docs.python.org/3/reference/datamodel.html#object.\_\_rand\_\_)
* [\_\_rfloordiv\_\_](https://docs.python.org/3/reference/datamodel.html#object.\_\_rfloordiv\_\_)
* [\_\_rlshift\_\_](https://docs.python.org/3/reference/datamodel.html#object.\_\_rlshift\_\_)
* [\_\_rmatmul\_\_](https://docs.python.org/3/reference/datamodel.html#object.\_\_rmatmul\_\_)
* [\_\_rmod\_\_](https://docs.python.org/3/reference/datamodel.html#object.\_\_rmod\_\_)
* [\_\_rmul\_\_](https://docs.python.org/3/reference/datamodel.html#object.\_\_rmul\_\_)
* [\_\_ror\_\_](https://docs.python.org/3/reference/datamodel.html#object.\_\_ror\_\_)
* [\_\_round\_\_](https://docs.python.org/3/reference/datamodel.html#object.\_\_round\_\_)
* [\_\_rpow\_\_](https://docs.python.org/3/reference/datamodel.html#object.\_\_rpow\_\_)
* [\_\_rrshift\_\_](https://docs.python.org/3/reference/datamodel.html#object.\_\_rrshift\_\_)
* [\_\_rshift\_\_](https://docs.python.org/3/reference/datamodel.html#object.\_\_rshift\_\_)
* [\_\_rsub\_\_](https://docs.python.org/3/reference/datamodel.html#object.\_\_rsub\_\_)
* [\_\_rtruediv\_\_](https://docs.python.org/3/reference/datamodel.html#object.\_\_rtruediv\_\_)
* [\_\_rxor\_\_](https://docs.python.org/3/reference/datamodel.html#object.\_\_rxor\_\_)
* [\_\_sub\_\_](https://docs.python.org/3/reference/datamodel.html#object.\_\_sub\_\_)
* [\_\_truediv\_\_](https://docs.python.org/3/reference/datamodel.html#object.\_\_truediv\_\_)
* [\_\_xor\_\_](https://docs.python.org/3/reference/datamodel.html#object.\_\_xor\_\_)

### Supported NumPy functions.

* [np.absolute](https://numpy.org/doc/stable/reference/generated/numpy.absolute.html)
* [np.add](https://numpy.org/doc/stable/reference/generated/numpy.add.html)
* [np.arccos](https://numpy.org/doc/stable/reference/generated/numpy.arccos.html)
* [np.arccosh](https://numpy.org/doc/stable/reference/generated/numpy.arccosh.html)
* [np.arcsin](https://numpy.org/doc/stable/reference/generated/numpy.arcsin.html)
* [np.arcsinh](https://numpy.org/doc/stable/reference/generated/numpy.arcsinh.html)
* [np.arctan](https://numpy.org/doc/stable/reference/generated/numpy.arctan.html)
* [np.arctan2](https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html)
* [np.arctanh](https://numpy.org/doc/stable/reference/generated/numpy.arctanh.html)
* [np.around](https://numpy.org/doc/stable/reference/generated/numpy.around.html)
* [np.bitwise\_and](https://numpy.org/doc/stable/reference/generated/numpy.bitwise\_and.html)
* [np.bitwise\_or](https://numpy.org/doc/stable/reference/generated/numpy.bitwise\_or.html)
* [np.bitwise\_xor](https://numpy.org/doc/stable/reference/generated/numpy.bitwise\_xor.html)
* [np.broadcast\_to](https://numpy.org/doc/stable/reference/generated/numpy.broadcast\_to.html)
* [np.cbrt](https://numpy.org/doc/stable/reference/generated/numpy.cbrt.html)
* [np.ceil](https://numpy.org/doc/stable/reference/generated/numpy.ceil.html)
* [np.clip](https://numpy.org/doc/stable/reference/generated/numpy.clip.html)
* [np.concatenate](https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html)
* [np.copysign](https://numpy.org/doc/stable/reference/generated/numpy.copysign.html)
* [np.cos](https://numpy.org/doc/stable/reference/generated/numpy.cos.html)
* [np.cosh](https://numpy.org/doc/stable/reference/generated/numpy.cosh.html)
* [np.deg2rad](https://numpy.org/doc/stable/reference/generated/numpy.deg2rad.html)
* [np.degrees](https://numpy.org/doc/stable/reference/generated/numpy.degrees.html)
* [np.dot](https://numpy.org/doc/stable/reference/generated/numpy.dot.html)
* [np.equal](https://numpy.org/doc/stable/reference/generated/numpy.equal.html)
* [np.exp](https://numpy.org/doc/stable/reference/generated/numpy.exp.html)
* [np.exp2](https://numpy.org/doc/stable/reference/generated/numpy.exp2.html)
* [np.expand\_dims](https://numpy.org/doc/stable/reference/generated/numpy.expand\_dims.html)
* [np.expm1](https://numpy.org/doc/stable/reference/generated/numpy.expm1.html)
* [np.fabs](https://numpy.org/doc/stable/reference/generated/numpy.fabs.html)
* [np.float\_power](https://numpy.org/doc/stable/reference/generated/numpy.float\_power.html)
* [np.floor](https://numpy.org/doc/stable/reference/generated/numpy.floor.html)
* [np.floor\_divide](https://numpy.org/doc/stable/reference/generated/numpy.floor\_divide.html)
* [np.fmax](https://numpy.org/doc/stable/reference/generated/numpy.fmax.html)
* [np.fmin](https://numpy.org/doc/stable/reference/generated/numpy.fmin.html)
* [np.fmod](https://numpy.org/doc/stable/reference/generated/numpy.fmod.html)
* [np.gcd](https://numpy.org/doc/stable/reference/generated/numpy.gcd.html)
* [np.greater](https://numpy.org/doc/stable/reference/generated/numpy.greater.html)
* [np.greater\_equal](https://numpy.org/doc/stable/reference/generated/numpy.greater\_equal.html)
* [np.heaviside](https://numpy.org/doc/stable/reference/generated/numpy.heaviside.html)
* [np.hypot](https://numpy.org/doc/stable/reference/generated/numpy.hypot.html)
* [np.invert](https://numpy.org/doc/stable/reference/generated/numpy.invert.html)
* [np.isfinite](https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html)
* [np.isinf](https://numpy.org/doc/stable/reference/generated/numpy.isinf.html)
* [np.isnan](https://numpy.org/doc/stable/reference/generated/numpy.isnan.html)
* [np.lcm](https://numpy.org/doc/stable/reference/generated/numpy.lcm.html)
* [np.ldexp](https://numpy.org/doc/stable/reference/generated/numpy.ldexp.html)
* [np.left\_shift](https://numpy.org/doc/stable/reference/generated/numpy.left\_shift.html)
* [np.less](https://numpy.org/doc/stable/reference/generated/numpy.less.html)
* [np.less\_equal](https://numpy.org/doc/stable/reference/generated/numpy.less\_equal.html)
* [np.log](https://numpy.org/doc/stable/reference/generated/numpy.log.html)
* [np.log10](https://numpy.org/doc/stable/reference/generated/numpy.log10.html)
* [np.log1p](https://numpy.org/doc/stable/reference/generated/numpy.log1p.html)
* [np.log2](https://numpy.org/doc/stable/reference/generated/numpy.log2.html)
* [np.logaddexp](https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html)
* [np.logaddexp2](https://numpy.org/doc/stable/reference/generated/numpy.logaddexp2.html)
* [np.logical\_and](https://numpy.org/doc/stable/reference/generated/numpy.logical\_and.html)
* [np.logical\_not](https://numpy.org/doc/stable/reference/generated/numpy.logical\_not.html)
* [np.logical\_or](https://numpy.org/doc/stable/reference/generated/numpy.logical\_or.html)
* [np.logical\_xor](https://numpy.org/doc/stable/reference/generated/numpy.logical\_xor.html)
* [np.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)
* [np.maximum](https://numpy.org/doc/stable/reference/generated/numpy.maximum.html)
* [np.minimum](https://numpy.org/doc/stable/reference/generated/numpy.minimum.html)
* [np.multiply](https://numpy.org/doc/stable/reference/generated/numpy.multiply.html)
* [np.negative](https://numpy.org/doc/stable/reference/generated/numpy.negative.html)
* [np.nextafter](https://numpy.org/doc/stable/reference/generated/numpy.nextafter.html)
* [np.not\_equal](https://numpy.org/doc/stable/reference/generated/numpy.not\_equal.html)
* [np.ones\_like](https://numpy.org/doc/stable/reference/generated/numpy.ones\_like.html)
* [np.positive](https://numpy.org/doc/stable/reference/generated/numpy.positive.html)
* [np.power](https://numpy.org/doc/stable/reference/generated/numpy.power.html)
* [np.rad2deg](https://numpy.org/doc/stable/reference/generated/numpy.rad2deg.html)
* [np.radians](https://numpy.org/doc/stable/reference/generated/numpy.radians.html)
* [np.reciprocal](https://numpy.org/doc/stable/reference/generated/numpy.reciprocal.html)
* [np.remainder](https://numpy.org/doc/stable/reference/generated/numpy.remainder.html)
* [np.reshape](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html)
* [np.right\_shift](https://numpy.org/doc/stable/reference/generated/numpy.right\_shift.html)
* [np.rint](https://numpy.org/doc/stable/reference/generated/numpy.rint.html)
* [np.round](https://numpy.org/doc/stable/reference/generated/numpy.round.html)
* [np.sign](https://numpy.org/doc/stable/reference/generated/numpy.sign.html)
* [np.signbit](https://numpy.org/doc/stable/reference/generated/numpy.signbit.html)
* [np.sin](https://numpy.org/doc/stable/reference/generated/numpy.sin.html)
* [np.sinh](https://numpy.org/doc/stable/reference/generated/numpy.sinh.html)
* [np.spacing](https://numpy.org/doc/stable/reference/generated/numpy.spacing.html)
* [np.sqrt](https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html)
* [np.square](https://numpy.org/doc/stable/reference/generated/numpy.square.html)
* [np.subtract](https://numpy.org/doc/stable/reference/generated/numpy.subtract.html)
* [np.sum](https://numpy.org/doc/stable/reference/generated/numpy.sum.html)
* [np.tan](https://numpy.org/doc/stable/reference/generated/numpy.tan.html)
* [np.tanh](https://numpy.org/doc/stable/reference/generated/numpy.tanh.html)
* [np.transpose](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html)
* [np.true\_divide](https://numpy.org/doc/stable/reference/generated/numpy.true\_divide.html)
* [np.trunc](https://numpy.org/doc/stable/reference/generated/numpy.trunc.html)
* [np.where](https://numpy.org/doc/stable/reference/generated/numpy.where.html)
* [np.zeros\_like](https://numpy.org/doc/stable/reference/generated/numpy.zeros\_like.html)

### Supported `ndarray` methods.

* [np.ndarray.astype](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html)
* [np.ndarray.clip](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.clip.html)
* [np.ndarray.dot](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.dot.html)
* [np.ndarray.flatten](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html)
* [np.ndarray.reshape](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.reshape.html)
* [np.ndarray.transpose](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.transpose.html)

### Supported `ndarray` properties.

* [np.ndarray.shape](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.shape.html)
* [np.ndarray.ndim](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ndim.html)
* [np.ndarray.size](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.size.html)
* [np.ndarray.T](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.T.html)

## Limitations

### Control flow constraints.

Some Python control flow statements are not supported. You cannot have an `if` statement or a `while` statement for which the condition depends on an encrypted value. However, such statements are supported with constant values (e.g., `for i in range(SOME_CONSTANT)`, `if os.environ.get("SOME_FEATURE") == "ON":`).

### Type constraints.

You cannot have floating-point inputs or floating-point outputs. You can have floating-point intermediate values as long as they can be converted to an integer Table Lookup (e.g., `(60 * np.sin(x)).astype(np.int64)`).

### Bit width constraints.

There is a limit on the bit width of encrypted values. We are constantly working on increasing this bit width. If you go above the limit, you will get an error.
