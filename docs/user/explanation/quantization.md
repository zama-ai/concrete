```{warning}
FIXME(Andrei): see if this is still appropriate, update etc; make link with use_quantization.md
```

# Quantization

```{note}
from [Wikipedia](https://en.wikipedia.org/wiki/Quantization):

> Quantization is the process of constraining an input from a continuous or otherwise large set of values (such as the real numbers) to a discrete set (such as the integers).
```

## Why is it needed?

Modern computing has long been using data types that use 32 or 64 bits (be that integers or floating point numbers), or even bigger data types. However due to the costly nature of FHE computations (see [the limits of FHE](fhe_and_framework_limits.md)), using such types with FHE is impractical (or plain impossible) to have computations executing in a reasonable amount of time.

## The gist of quantization

The basic idea of quantization is to take a range of values represented by a _large_ data type and represent it by a _smaller_ data type. This means some accuracy in the number's representation is lost, but in a lot of cases it is possible to adapt computations to still give meaningful results while using significantly less bits to sent the data used during those computations.

## Quantization in practice

Let's first define some notations. Let $ [\alpha, \beta ] $ be the range of our value to quantize where $ \alpha $ is the minimum and $ \beta $ is the maximum.

To quantize a range with floating point values (in $ \mathbb{R} $) to unsigned integer values (in $ \mathbb{N} $), we first need to choose the data type that is going to be used. **ConcreteLib**, the library used in the **Concrete Framework**, is currently limited to 7 bits unsigned integers, so we'll use that for the example. Knowing that, for a value in the range $ [\alpha, \beta ] $, we can compute the `scale` $ S $ of the quantization:

$$ S =  \frac{\beta - \alpha}{2^n - 1} $$


 where $ n $ is the number of bits (here 7). In practice the quantization scale is then $ S = \frac{\beta - \alpha}{127} $. This means the gap between consecutive representible values cannot be smaller than that $ S $ value which means there can be a substantial loss of precision. Every interval of length $ S $ will be represented by a value within the range $ [0..127] $.

The other important parameter from this quantization schema is the `zero point` $ Z $ value. This essentially brings the 0 floating point value to a specific integer. Doing this allows us to have an asymetric quantization where the resulting integer is in the unsigned integer realm, $ \mathbb{N} $. 

$$ Z = \mathsc{round} \left(- \frac{\alpha}{S} \right) $$

There is more mathematics involved in how computations change when replacing floating point values by integers for a fully connected or a convolution layer. The IntelLabs distiller quantization documentation goes into a [detailed explanation](https://intellabs.github.io/distiller/algo_quantization.html) about the maths to quantize values and how to keep computations consistent.

Regarding quantization and FHE compilation, it is important to understand the difference between two modes:

1. the quantization is done before the compilation; notably, the quantization is completely controlled by the user, and can be done by any means, including by using third party frameworks
2. the quantization is done during the compilation (inside our framework), with much less control by the user.

For the moment, only the second method is available in Concrete Framework, but we plan to have the first method available in a further release, since it should give more freedom and better results to the user.

We detail the use of quantization within Concrete Framework in [here](../howto/use_quantization.md).
## Resources

- IntelLabs distiller explanation of quantization: [Distiller documentation](https://intellabs.github.io/distiller/algo_quantization.html)
- Lei Mao's blog on quantization: [Quantization for Neural Networks](https://leimao.github.io/article/Neural-Networks-Quantization/)
- Google paper on Neural Network quantization and integer only inference: [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)
