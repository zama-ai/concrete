# Using Quantization in Concrete Framework

In this section we detail some usage of [quantization](../explanation/quantization.md) as implemented in Concrete.

## Quantization Basics

The very basic purpose of the quantization is to convert floating point values to integers. We can apply such conversion using `QuantizedArray` available in `concrete.quantization`. 

`QuantizedArray` takes 2 arguments:
- `n_bits` that defines the precision of the quantization. Currently, `n_bits` is limited to 7, due to some ConcreteLib limits.
- `values` that will be converted to integers

```python
from concrete.quantization import QuantizedArray
import numpy
numpy.random.seed(0)
A = numpy.random.uniform(-2, 2, 10)
# array([ 0.19525402,  0.86075747,  0.4110535,  0.17953273, -0.3053808,
#         0.58357645, -0.24965115,  1.567092 ,  1.85465104, -0.46623392])
q_A = QuantizedArray(7, A)
q_A.qvalues
# array([ 37,          73,          48,         36,          9,  
#         58,          12,          112,        127,         0])
# the quantized integers values from A.
q_A.scale
# 0.018274684777173276, the scale S.
q_A.zero_point 
# 26, the zero point Z.
q_A.dequant()
# array([ 0.20102153,  0.85891018,  0.40204307,  0.18274685, -0.31066964,
#         0.58478991, -0.25584559,  1.57162289,  1.84574316, -0.4751418 ])
# Dequantized values.
```

## Operations in the Quantized Realm

## Why are the scale and zero point not a problem ?

## Compilation

## Future releases