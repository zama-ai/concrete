# How Does It Work ?

The `concrete-boolean` crate uses `concrete-core`, a cryptographic library for [Fully
Homomorphic Encryption](https://en.wikipedia.org/wiki/Homomorphic_encryption) developed at
**Zama**.

In a nutshell, homomorphic encryption is a cryptographic paradigm enabling to **compute on
encrypted data**.

A scheme is said to be fully homomorphic when it is able to evaluate an arithmetic circuit (i.e.,
composed by additions and multiplications) of arbitrary depth.

The `concrete-core` crate contains each homomorphic operator needed to build boolean gate 
evaluation in a
stable manner. 
Built on top of `concrete-core`,  `concrete-boolean` can execute **boolean circuits of any 
length** in an encrypted
way.

You can find the description of the algorithms in this [TFHE paper](https://eprint.iacr.org/2018/421.pdf).

## How secure is it?

The cryptographic scheme used in the `concrete-core`, is a variant of [Regev 
cryptosystem](https://cims.nyu.edu/~regev/papers/lwesurvey.pdf), 
and is based on a problem so hard to solve, that even quantum computers can not break it easily.

In practice, you need to tune some cryptographic parameters, in order to ensure the correctness
of the result, and the security of the computation.

To make it simpler, **we provide two sets of parameters**, which ensuring correct computation for a
certain probability with the standard security of 128 bits:

+ `concrete_boolean::parameters::DEFAULT_PARAMETERS`
+ `concrete_boolean::parameters::TFHE_LIB_PARAMETERS`

Note that if you desire, you can also create your own set of parameters.
This is an `unsafe` operation as failing to properly fix the parameters will potentially result
with an incorrect and/or unsecure computation:

```rust
extern crate concrete_boolean;
extern crate concrete_commons;
use concrete_boolean::parameters::BooleanParameters;
use concrete_commons::parameters::*;
use concrete_commons::dispersion::*;

// WARNING: might be unsecure and/or incorrect
// You can create your own set of parameters
let parameters = unsafe{
    BooleanParameters::new_unsecure(
        LweDimension(586),
        GlweDimension(2),
        PolynomialSize(512),
        StandardDev(0.00008976167396834998),
        StandardDev(0.00000002989040792967434),
        DecompositionBaseLog(8),
        DecompositionLevelCount(2),
        DecompositionBaseLog(2),
        DecompositionLevelCount(5),
    )
};
```
