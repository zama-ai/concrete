```{warning}
FIXME(Jordan/Andrei): see if this is still appropriate, update etc; make link with USE_QUANTIZATION.md
```

# Quantization

```{note}
from [Wikipedia](https://en.wikipedia.org/wiki/Quantization):

> Quantization is the process of constraining an input from a continuous or otherwise large set of values (such as the real numbers) to a discrete set (such as the integers).
```

## Why is it needed?

Modern computing has long been using data types that use 32 or 64 bits (be that integers or floating point numbers), or even bigger data types. However due to the costly nature of FHE computations (see [the limits of FHE](FHE_AND_FRAMEWORK_LIMITS.md)), using such types with FHE is impractical (or plain impossible) to have computations executing in a reasonable amount of time.

## The gist of quantization

The basic idea of quantization is to take a range of values represented by a _large_ data type and represent it by a _smaller_ data type. This means some accuracy in the number's representation is lost, but in a lot of cases it is possible to adapt computations to still give meaningful results while using significantly less bits to sent the data used during those computations.

## Quantization in practice

To quantize a range of values on a smaller range of values, we first need to choose the data type that is going to be used. **ConcreteLib**, the library used in the **Concrete Framework**, is currently limited to 7 bits unsigned integers, so we'll use that for the example. Knowing that, for a value in the range `[min_range, max_range]`, we can compute the step of the quantization, which is `(max_range - min_range) / (2**n - 1)` where n is the number of bits, here 7, so in practice the quantization step is `step = (max_range - min_range) / 127`. This means the gap between consecutive representible values cannot be smaller than that `step` value which means there can be a substantial loss of precision. Every interval of length `step = (max_range - min_range) / 127` will be represented by a value in `[0..127]`.

The IntelLabs distiller quantization documentation goes into a detailed explanation about the math to quantize values and how to keep computations consistent: [quantization algorithm documentation](https://intellabs.github.io/distiller/algo_quantization.html).

## Resources

- IntelLabs distiller explanation of quantization: [Distiller documentation](https://intellabs.github.io/distiller/algo_quantization.html)
- Lei Mao's blog on quantization: [Quantization for Neural Networks](https://leimao.github.io/article/Neural-Networks-Quantization/)
- Google paper on Neural Network quantization and integer only inference: [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)
