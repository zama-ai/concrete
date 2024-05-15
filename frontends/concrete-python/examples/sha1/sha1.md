# SHA1 Computation with Modules

We take the example of a SHA1 computation to show how
[Modules](https://docs.zama.ai/concrete/compilation/modules) can be useful in Concrete. SHA1 is a
deprecated broken hash function: we didn't take this example for its security, but rather as it's a
pedagogical example of how Modules can be useful.

The SHA1 code is given [here](sha1.py). Execution times of the different functions are given in the
final section.

We made our example by forking
[https://github.com/ajalt/python-sha1](https://github.com/ajalt/python-sha1) by
[AJ Alt](https://github.com/ajalt), and then made lot of modifications to have an FHE implementation
and corresponding tests.

## SHA1 Overview

SHA1 is a deprecated broken hash function defined by NIST in 1995: one can find information about it
in its [Wikipedia page](https://en.wikipedia.org/wiki/SHA-1), or in its
[official description](https://web.archive.org/web/20161126003357/http://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf).

Its [pseudo-code](https://en.wikipedia.org/wiki/SHA-1#SHA-1_pseudocode) is particularly interesting,
since our implementation in Concrete Python will follow the structure.

## Our FHE Implementation

In [our implementation](sha1.py), only the compression function is done in FHE. It corresponds to the
`_process_encrypted_chunk_server_side` function. The rest, including the message expansion is done
client side in the clear. It would be possible to do more in FHE, but it was not needed for the
purpose of this tutorial, which is about showing the use of
[Modules](https://docs.zama.ai/concrete/compilation/modules).

Our Module is:

```
@fhe.module()
class MyModule:
    @fhe.function({"x": "encrypted", "y": "encrypted", "z": "encrypted"})
    def xor3(x, y, z):
        return x ^ y ^ z

    @fhe.function({"x": "encrypted", "y": "encrypted", "z": "encrypted"})
    def iftern(x, y, z):
        return z ^ (x & (y ^ z))

    @fhe.function({"x": "encrypted", "y": "encrypted", "z": "encrypted"})
    def maj(x, y, z):
        return (x & y) | (z & (x | y))

    @fhe.function({"x": "encrypted"})
    def rotate30(x):
        ans = fhe.zeros((32,))
        ans[30:32] = x[0:2]
        ans[0:30] = x[2:32]
        return ans

    @fhe.function({"x": "encrypted"})
    def rotate5(x):
        ans = fhe.zeros((32,))
        ans[5:32] = x[0:27]
        ans[0:5] = x[27:32]
        return ans

    @fhe.function({"x": "encrypted", "y": "encrypted"})
    def add2(x, y):
        ans = fhe.zeros((32,))
        cy = 0

        for i in range(32):
            t = x[i] + y[i] + cy
            cy, tr = t >= 2, t % 2
            ans[i] = tr

        return ans

    @fhe.function(
        {"x": "encrypted", "y": "encrypted", "u": "encrypted", "v": "encrypted", "w": "encrypted"}
    )
    def add5(x, y, u, v, w):
        ans = fhe.zeros((32,))
        cy = 0

        for i in range(32):
            t = x[i] + y[i] + cy
            cy, tr = t // 2, t % 2
            ans[i] = tr

        cy = 0

        for i in range(32):
            t = ans[i] + u[i] + cy
            cy, tr = t // 2, t % 2
            ans[i] = tr

        cy = 0

        for i in range(32):
            t = ans[i] + v[i] + cy
            cy, tr = t // 2, t % 2
            ans[i] = tr

        cy = 0

        for i in range(32):
            t = ans[i] + w[i] + cy
            cy, tr = t // 2, t % 2
            ans[i] = tr

        return ans
```

It contains 7 functions which can be combined together:
- `xor3` XORs three values together
- `iftern` computes the ternary IF, i.e., c ? t : f
- `maj` computes the majority function
- `rotate30` rotates a 32-bit word by 30 positions
- `rotate5` rotates a 32-bit word by 5 positions
- `add2` adds two values together
- `add5` adds five values together

This module is then compiled. Remark we set `p_error=10**-8` to a very small value, to avoid
errors in computations. Thanks to the Module computation, all these functions can be combined
together, i.e., outputs of some of them can be reused as inputs of some other ones. It is very
convenient to have a larger function, with some control-flow (conditions, branches, loops) done in
the clear and operating with these small functions. In our case, it's done in
`_process_encrypted_chunk_server_side` function.

## Details of `_process_encrypted_chunk_server_side`

`_process_encrypted_chunk_server_side` uses inputs which are encrypted, and returns encrypted
values. Everything is a 32-bit word when done in the clear. Here, words will be represented as 32
1-bit encrypted values, to simplify (and speed) the non-linear operations in SHA1.

Then, we have the main loop of the compression function which is done in the clear with
`for i in range(80):`. There, we also have the different conditions (`if 0 <= i <= 19:`,
`elif 20 <= i <= 39` etc) which is done in the clear. Obviously, there is no security issue here,
since the `i` index is not secret. Depending on `i`, we then select the right `fsplit_enc` and
`ksplit`, which correspond to `f` and `k` in the clear implementation of the compression function.
Here, we can see a first use of the different functions in the Module.

Then, we continue with other functions in the module, to compute `arot5 = _left_rotate(a, 5)` and
`arot5 + f + e + k + w[i]`. Finally, we update the different `a, b, c, d, e` values as in the clear
implementation but with the encrypted forms.

One can remark that we have compiled the different functions of the Module on inputset made with
bits. Under the hood, Concrete will add a few programmable bootstrapping, to compute the right
functions in FHE.

## MLIR code

Compiling with `show_mlir = True` allows to see the different MLIR implementations. Typically, with
current Concrete version, it would give something like:

<<<<<<< HEAD
=======
- `add2` function
<details>
>>>>>>> 78decb21 (docs(frontend): adding a SHA1 tutorial with modules)
```
  func.func @add2(%arg0: tensor<32x!FHE.eint<2>>, %arg1: tensor<32x!FHE.eint<2>>) -> tensor<32x!FHE.eint<2>> {
    %0 = "FHE.zero_tensor"() : () -> tensor<32x!FHE.eint<2>>
    %c0 = arith.constant 0 : index
    %extracted = tensor.extract %arg0[%c0] : tensor<32x!FHE.eint<2>>
    %extracted_0 = tensor.extract %arg1[%c0] : tensor<32x!FHE.eint<2>>
    %1 = "FHE.add_eint"(%extracted, %extracted_0) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %c0_i2 = arith.constant 0 : i2
    %2 = "FHE.add_eint_int"(%1, %c0_i2) : (!FHE.eint<2>, i2) -> !FHE.eint<2>
    %c2_i3 = arith.constant 2 : i3
    %cst = arith.constant dense<[0, 0, 1, 1]> : tensor<4xi64>
    %3 = "FHE.apply_lookup_table"(%2, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %cst_1 = arith.constant dense<[0, 1, 0, 1]> : tensor<4xi64>
    %4 = "FHE.apply_lookup_table"(%2, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements = tensor.from_elements %4 : tensor<1x!FHE.eint<2>>
    %inserted_slice = tensor.insert_slice %from_elements into %0[0] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c1 = arith.constant 1 : index
    %extracted_2 = tensor.extract %arg0[%c1] : tensor<32x!FHE.eint<2>>
    %extracted_3 = tensor.extract %arg1[%c1] : tensor<32x!FHE.eint<2>>
    %5 = "FHE.add_eint"(%extracted_2, %extracted_3) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %6 = "FHE.add_eint"(%5, %3) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %7 = "FHE.apply_lookup_table"(%6, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %8 = "FHE.apply_lookup_table"(%6, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_4 = tensor.from_elements %8 : tensor<1x!FHE.eint<2>>
    %inserted_slice_5 = tensor.insert_slice %from_elements_4 into %inserted_slice[1] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c2 = arith.constant 2 : index
    %extracted_6 = tensor.extract %arg0[%c2] : tensor<32x!FHE.eint<2>>
    %extracted_7 = tensor.extract %arg1[%c2] : tensor<32x!FHE.eint<2>>
    %9 = "FHE.add_eint"(%extracted_6, %extracted_7) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %10 = "FHE.add_eint"(%9, %7) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %11 = "FHE.apply_lookup_table"(%10, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %12 = "FHE.apply_lookup_table"(%10, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_8 = tensor.from_elements %12 : tensor<1x!FHE.eint<2>>
    %inserted_slice_9 = tensor.insert_slice %from_elements_8 into %inserted_slice_5[2] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c3 = arith.constant 3 : index
    %extracted_10 = tensor.extract %arg0[%c3] : tensor<32x!FHE.eint<2>>
    %extracted_11 = tensor.extract %arg1[%c3] : tensor<32x!FHE.eint<2>>
    %13 = "FHE.add_eint"(%extracted_10, %extracted_11) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %14 = "FHE.add_eint"(%13, %11) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %15 = "FHE.apply_lookup_table"(%14, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %16 = "FHE.apply_lookup_table"(%14, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_12 = tensor.from_elements %16 : tensor<1x!FHE.eint<2>>
    %inserted_slice_13 = tensor.insert_slice %from_elements_12 into %inserted_slice_9[3] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c4 = arith.constant 4 : index
    %extracted_14 = tensor.extract %arg0[%c4] : tensor<32x!FHE.eint<2>>
    %extracted_15 = tensor.extract %arg1[%c4] : tensor<32x!FHE.eint<2>>
    %17 = "FHE.add_eint"(%extracted_14, %extracted_15) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %18 = "FHE.add_eint"(%17, %15) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %19 = "FHE.apply_lookup_table"(%18, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %20 = "FHE.apply_lookup_table"(%18, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_16 = tensor.from_elements %20 : tensor<1x!FHE.eint<2>>
    %inserted_slice_17 = tensor.insert_slice %from_elements_16 into %inserted_slice_13[4] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c5 = arith.constant 5 : index
    %extracted_18 = tensor.extract %arg0[%c5] : tensor<32x!FHE.eint<2>>
    %extracted_19 = tensor.extract %arg1[%c5] : tensor<32x!FHE.eint<2>>
    %21 = "FHE.add_eint"(%extracted_18, %extracted_19) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %22 = "FHE.add_eint"(%21, %19) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %23 = "FHE.apply_lookup_table"(%22, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %24 = "FHE.apply_lookup_table"(%22, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_20 = tensor.from_elements %24 : tensor<1x!FHE.eint<2>>
    %inserted_slice_21 = tensor.insert_slice %from_elements_20 into %inserted_slice_17[5] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c6 = arith.constant 6 : index
    %extracted_22 = tensor.extract %arg0[%c6] : tensor<32x!FHE.eint<2>>
    %extracted_23 = tensor.extract %arg1[%c6] : tensor<32x!FHE.eint<2>>
    %25 = "FHE.add_eint"(%extracted_22, %extracted_23) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %26 = "FHE.add_eint"(%25, %23) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %27 = "FHE.apply_lookup_table"(%26, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %28 = "FHE.apply_lookup_table"(%26, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_24 = tensor.from_elements %28 : tensor<1x!FHE.eint<2>>
    %inserted_slice_25 = tensor.insert_slice %from_elements_24 into %inserted_slice_21[6] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c7 = arith.constant 7 : index
    %extracted_26 = tensor.extract %arg0[%c7] : tensor<32x!FHE.eint<2>>
    %extracted_27 = tensor.extract %arg1[%c7] : tensor<32x!FHE.eint<2>>
    %29 = "FHE.add_eint"(%extracted_26, %extracted_27) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %30 = "FHE.add_eint"(%29, %27) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %31 = "FHE.apply_lookup_table"(%30, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %32 = "FHE.apply_lookup_table"(%30, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_28 = tensor.from_elements %32 : tensor<1x!FHE.eint<2>>
    %inserted_slice_29 = tensor.insert_slice %from_elements_28 into %inserted_slice_25[7] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c8 = arith.constant 8 : index
    %extracted_30 = tensor.extract %arg0[%c8] : tensor<32x!FHE.eint<2>>
    %extracted_31 = tensor.extract %arg1[%c8] : tensor<32x!FHE.eint<2>>
    %33 = "FHE.add_eint"(%extracted_30, %extracted_31) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %34 = "FHE.add_eint"(%33, %31) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %35 = "FHE.apply_lookup_table"(%34, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %36 = "FHE.apply_lookup_table"(%34, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_32 = tensor.from_elements %36 : tensor<1x!FHE.eint<2>>
    %inserted_slice_33 = tensor.insert_slice %from_elements_32 into %inserted_slice_29[8] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c9 = arith.constant 9 : index
    %extracted_34 = tensor.extract %arg0[%c9] : tensor<32x!FHE.eint<2>>
    %extracted_35 = tensor.extract %arg1[%c9] : tensor<32x!FHE.eint<2>>
    %37 = "FHE.add_eint"(%extracted_34, %extracted_35) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %38 = "FHE.add_eint"(%37, %35) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %39 = "FHE.apply_lookup_table"(%38, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %40 = "FHE.apply_lookup_table"(%38, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_36 = tensor.from_elements %40 : tensor<1x!FHE.eint<2>>
    %inserted_slice_37 = tensor.insert_slice %from_elements_36 into %inserted_slice_33[9] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c10 = arith.constant 10 : index
    %extracted_38 = tensor.extract %arg0[%c10] : tensor<32x!FHE.eint<2>>
    %extracted_39 = tensor.extract %arg1[%c10] : tensor<32x!FHE.eint<2>>
    %41 = "FHE.add_eint"(%extracted_38, %extracted_39) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %42 = "FHE.add_eint"(%41, %39) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %43 = "FHE.apply_lookup_table"(%42, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %44 = "FHE.apply_lookup_table"(%42, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_40 = tensor.from_elements %44 : tensor<1x!FHE.eint<2>>
    %inserted_slice_41 = tensor.insert_slice %from_elements_40 into %inserted_slice_37[10] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c11 = arith.constant 11 : index
    %extracted_42 = tensor.extract %arg0[%c11] : tensor<32x!FHE.eint<2>>
    %extracted_43 = tensor.extract %arg1[%c11] : tensor<32x!FHE.eint<2>>
    %45 = "FHE.add_eint"(%extracted_42, %extracted_43) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %46 = "FHE.add_eint"(%45, %43) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %47 = "FHE.apply_lookup_table"(%46, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %48 = "FHE.apply_lookup_table"(%46, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_44 = tensor.from_elements %48 : tensor<1x!FHE.eint<2>>
    %inserted_slice_45 = tensor.insert_slice %from_elements_44 into %inserted_slice_41[11] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c12 = arith.constant 12 : index
    %extracted_46 = tensor.extract %arg0[%c12] : tensor<32x!FHE.eint<2>>
    %extracted_47 = tensor.extract %arg1[%c12] : tensor<32x!FHE.eint<2>>
    %49 = "FHE.add_eint"(%extracted_46, %extracted_47) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %50 = "FHE.add_eint"(%49, %47) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %51 = "FHE.apply_lookup_table"(%50, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %52 = "FHE.apply_lookup_table"(%50, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_48 = tensor.from_elements %52 : tensor<1x!FHE.eint<2>>
    %inserted_slice_49 = tensor.insert_slice %from_elements_48 into %inserted_slice_45[12] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c13 = arith.constant 13 : index
    %extracted_50 = tensor.extract %arg0[%c13] : tensor<32x!FHE.eint<2>>
    %extracted_51 = tensor.extract %arg1[%c13] : tensor<32x!FHE.eint<2>>
    %53 = "FHE.add_eint"(%extracted_50, %extracted_51) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %54 = "FHE.add_eint"(%53, %51) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %55 = "FHE.apply_lookup_table"(%54, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %56 = "FHE.apply_lookup_table"(%54, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_52 = tensor.from_elements %56 : tensor<1x!FHE.eint<2>>
    %inserted_slice_53 = tensor.insert_slice %from_elements_52 into %inserted_slice_49[13] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c14 = arith.constant 14 : index
    %extracted_54 = tensor.extract %arg0[%c14] : tensor<32x!FHE.eint<2>>
    %extracted_55 = tensor.extract %arg1[%c14] : tensor<32x!FHE.eint<2>>
    %57 = "FHE.add_eint"(%extracted_54, %extracted_55) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %58 = "FHE.add_eint"(%57, %55) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %59 = "FHE.apply_lookup_table"(%58, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %60 = "FHE.apply_lookup_table"(%58, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_56 = tensor.from_elements %60 : tensor<1x!FHE.eint<2>>
    %inserted_slice_57 = tensor.insert_slice %from_elements_56 into %inserted_slice_53[14] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c15 = arith.constant 15 : index
    %extracted_58 = tensor.extract %arg0[%c15] : tensor<32x!FHE.eint<2>>
    %extracted_59 = tensor.extract %arg1[%c15] : tensor<32x!FHE.eint<2>>
    %61 = "FHE.add_eint"(%extracted_58, %extracted_59) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %62 = "FHE.add_eint"(%61, %59) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %63 = "FHE.apply_lookup_table"(%62, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %64 = "FHE.apply_lookup_table"(%62, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_60 = tensor.from_elements %64 : tensor<1x!FHE.eint<2>>
    %inserted_slice_61 = tensor.insert_slice %from_elements_60 into %inserted_slice_57[15] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c16 = arith.constant 16 : index
    %extracted_62 = tensor.extract %arg0[%c16] : tensor<32x!FHE.eint<2>>
    %extracted_63 = tensor.extract %arg1[%c16] : tensor<32x!FHE.eint<2>>
    %65 = "FHE.add_eint"(%extracted_62, %extracted_63) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %66 = "FHE.add_eint"(%65, %63) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %67 = "FHE.apply_lookup_table"(%66, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %68 = "FHE.apply_lookup_table"(%66, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_64 = tensor.from_elements %68 : tensor<1x!FHE.eint<2>>
    %inserted_slice_65 = tensor.insert_slice %from_elements_64 into %inserted_slice_61[16] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c17 = arith.constant 17 : index
    %extracted_66 = tensor.extract %arg0[%c17] : tensor<32x!FHE.eint<2>>
    %extracted_67 = tensor.extract %arg1[%c17] : tensor<32x!FHE.eint<2>>
    %69 = "FHE.add_eint"(%extracted_66, %extracted_67) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %70 = "FHE.add_eint"(%69, %67) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %71 = "FHE.apply_lookup_table"(%70, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %72 = "FHE.apply_lookup_table"(%70, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_68 = tensor.from_elements %72 : tensor<1x!FHE.eint<2>>
    %inserted_slice_69 = tensor.insert_slice %from_elements_68 into %inserted_slice_65[17] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c18 = arith.constant 18 : index
    %extracted_70 = tensor.extract %arg0[%c18] : tensor<32x!FHE.eint<2>>
    %extracted_71 = tensor.extract %arg1[%c18] : tensor<32x!FHE.eint<2>>
    %73 = "FHE.add_eint"(%extracted_70, %extracted_71) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %74 = "FHE.add_eint"(%73, %71) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %75 = "FHE.apply_lookup_table"(%74, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %76 = "FHE.apply_lookup_table"(%74, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_72 = tensor.from_elements %76 : tensor<1x!FHE.eint<2>>
    %inserted_slice_73 = tensor.insert_slice %from_elements_72 into %inserted_slice_69[18] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c19 = arith.constant 19 : index
    %extracted_74 = tensor.extract %arg0[%c19] : tensor<32x!FHE.eint<2>>
    %extracted_75 = tensor.extract %arg1[%c19] : tensor<32x!FHE.eint<2>>
    %77 = "FHE.add_eint"(%extracted_74, %extracted_75) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %78 = "FHE.add_eint"(%77, %75) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %79 = "FHE.apply_lookup_table"(%78, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %80 = "FHE.apply_lookup_table"(%78, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_76 = tensor.from_elements %80 : tensor<1x!FHE.eint<2>>
    %inserted_slice_77 = tensor.insert_slice %from_elements_76 into %inserted_slice_73[19] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c20 = arith.constant 20 : index
    %extracted_78 = tensor.extract %arg0[%c20] : tensor<32x!FHE.eint<2>>
    %extracted_79 = tensor.extract %arg1[%c20] : tensor<32x!FHE.eint<2>>
    %81 = "FHE.add_eint"(%extracted_78, %extracted_79) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %82 = "FHE.add_eint"(%81, %79) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %83 = "FHE.apply_lookup_table"(%82, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %84 = "FHE.apply_lookup_table"(%82, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_80 = tensor.from_elements %84 : tensor<1x!FHE.eint<2>>
    %inserted_slice_81 = tensor.insert_slice %from_elements_80 into %inserted_slice_77[20] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c21 = arith.constant 21 : index
    %extracted_82 = tensor.extract %arg0[%c21] : tensor<32x!FHE.eint<2>>
    %extracted_83 = tensor.extract %arg1[%c21] : tensor<32x!FHE.eint<2>>
    %85 = "FHE.add_eint"(%extracted_82, %extracted_83) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %86 = "FHE.add_eint"(%85, %83) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %87 = "FHE.apply_lookup_table"(%86, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %88 = "FHE.apply_lookup_table"(%86, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_84 = tensor.from_elements %88 : tensor<1x!FHE.eint<2>>
    %inserted_slice_85 = tensor.insert_slice %from_elements_84 into %inserted_slice_81[21] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c22 = arith.constant 22 : index
    %extracted_86 = tensor.extract %arg0[%c22] : tensor<32x!FHE.eint<2>>
    %extracted_87 = tensor.extract %arg1[%c22] : tensor<32x!FHE.eint<2>>
    %89 = "FHE.add_eint"(%extracted_86, %extracted_87) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %90 = "FHE.add_eint"(%89, %87) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %91 = "FHE.apply_lookup_table"(%90, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %92 = "FHE.apply_lookup_table"(%90, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_88 = tensor.from_elements %92 : tensor<1x!FHE.eint<2>>
    %inserted_slice_89 = tensor.insert_slice %from_elements_88 into %inserted_slice_85[22] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c23 = arith.constant 23 : index
    %extracted_90 = tensor.extract %arg0[%c23] : tensor<32x!FHE.eint<2>>
    %extracted_91 = tensor.extract %arg1[%c23] : tensor<32x!FHE.eint<2>>
    %93 = "FHE.add_eint"(%extracted_90, %extracted_91) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %94 = "FHE.add_eint"(%93, %91) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %95 = "FHE.apply_lookup_table"(%94, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %96 = "FHE.apply_lookup_table"(%94, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_92 = tensor.from_elements %96 : tensor<1x!FHE.eint<2>>
    %inserted_slice_93 = tensor.insert_slice %from_elements_92 into %inserted_slice_89[23] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c24 = arith.constant 24 : index
    %extracted_94 = tensor.extract %arg0[%c24] : tensor<32x!FHE.eint<2>>
    %extracted_95 = tensor.extract %arg1[%c24] : tensor<32x!FHE.eint<2>>
    %97 = "FHE.add_eint"(%extracted_94, %extracted_95) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %98 = "FHE.add_eint"(%97, %95) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %99 = "FHE.apply_lookup_table"(%98, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %100 = "FHE.apply_lookup_table"(%98, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_96 = tensor.from_elements %100 : tensor<1x!FHE.eint<2>>
    %inserted_slice_97 = tensor.insert_slice %from_elements_96 into %inserted_slice_93[24] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c25 = arith.constant 25 : index
    %extracted_98 = tensor.extract %arg0[%c25] : tensor<32x!FHE.eint<2>>
    %extracted_99 = tensor.extract %arg1[%c25] : tensor<32x!FHE.eint<2>>
    %101 = "FHE.add_eint"(%extracted_98, %extracted_99) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %102 = "FHE.add_eint"(%101, %99) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %103 = "FHE.apply_lookup_table"(%102, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %104 = "FHE.apply_lookup_table"(%102, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_100 = tensor.from_elements %104 : tensor<1x!FHE.eint<2>>
    %inserted_slice_101 = tensor.insert_slice %from_elements_100 into %inserted_slice_97[25] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c26 = arith.constant 26 : index
    %extracted_102 = tensor.extract %arg0[%c26] : tensor<32x!FHE.eint<2>>
    %extracted_103 = tensor.extract %arg1[%c26] : tensor<32x!FHE.eint<2>>
    %105 = "FHE.add_eint"(%extracted_102, %extracted_103) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %106 = "FHE.add_eint"(%105, %103) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %107 = "FHE.apply_lookup_table"(%106, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %108 = "FHE.apply_lookup_table"(%106, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_104 = tensor.from_elements %108 : tensor<1x!FHE.eint<2>>
    %inserted_slice_105 = tensor.insert_slice %from_elements_104 into %inserted_slice_101[26] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c27 = arith.constant 27 : index
    %extracted_106 = tensor.extract %arg0[%c27] : tensor<32x!FHE.eint<2>>
    %extracted_107 = tensor.extract %arg1[%c27] : tensor<32x!FHE.eint<2>>
    %109 = "FHE.add_eint"(%extracted_106, %extracted_107) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %110 = "FHE.add_eint"(%109, %107) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %111 = "FHE.apply_lookup_table"(%110, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %112 = "FHE.apply_lookup_table"(%110, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_108 = tensor.from_elements %112 : tensor<1x!FHE.eint<2>>
    %inserted_slice_109 = tensor.insert_slice %from_elements_108 into %inserted_slice_105[27] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c28 = arith.constant 28 : index
    %extracted_110 = tensor.extract %arg0[%c28] : tensor<32x!FHE.eint<2>>
    %extracted_111 = tensor.extract %arg1[%c28] : tensor<32x!FHE.eint<2>>
    %113 = "FHE.add_eint"(%extracted_110, %extracted_111) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %114 = "FHE.add_eint"(%113, %111) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %115 = "FHE.apply_lookup_table"(%114, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %116 = "FHE.apply_lookup_table"(%114, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_112 = tensor.from_elements %116 : tensor<1x!FHE.eint<2>>
    %inserted_slice_113 = tensor.insert_slice %from_elements_112 into %inserted_slice_109[28] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c29 = arith.constant 29 : index
    %extracted_114 = tensor.extract %arg0[%c29] : tensor<32x!FHE.eint<2>>
    %extracted_115 = tensor.extract %arg1[%c29] : tensor<32x!FHE.eint<2>>
    %117 = "FHE.add_eint"(%extracted_114, %extracted_115) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %118 = "FHE.add_eint"(%117, %115) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %119 = "FHE.apply_lookup_table"(%118, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %120 = "FHE.apply_lookup_table"(%118, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_116 = tensor.from_elements %120 : tensor<1x!FHE.eint<2>>
    %inserted_slice_117 = tensor.insert_slice %from_elements_116 into %inserted_slice_113[29] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c30 = arith.constant 30 : index
    %extracted_118 = tensor.extract %arg0[%c30] : tensor<32x!FHE.eint<2>>
    %extracted_119 = tensor.extract %arg1[%c30] : tensor<32x!FHE.eint<2>>
    %121 = "FHE.add_eint"(%extracted_118, %extracted_119) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %122 = "FHE.add_eint"(%121, %119) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %123 = "FHE.apply_lookup_table"(%122, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %124 = "FHE.apply_lookup_table"(%122, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_120 = tensor.from_elements %124 : tensor<1x!FHE.eint<2>>
    %inserted_slice_121 = tensor.insert_slice %from_elements_120 into %inserted_slice_117[30] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c31 = arith.constant 31 : index
    %extracted_122 = tensor.extract %arg0[%c31] : tensor<32x!FHE.eint<2>>
    %extracted_123 = tensor.extract %arg1[%c31] : tensor<32x!FHE.eint<2>>
    %125 = "FHE.add_eint"(%extracted_122, %extracted_123) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %126 = "FHE.add_eint"(%125, %123) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %127 = "FHE.apply_lookup_table"(%126, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_124 = tensor.from_elements %127 : tensor<1x!FHE.eint<2>>
    %inserted_slice_125 = tensor.insert_slice %from_elements_124 into %inserted_slice_121[31] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    return %inserted_slice_125 : tensor<32x!FHE.eint<2>>
  }
```
<<<<<<< HEAD

=======
</details>

- `add5` function
<details>
>>>>>>> 78decb21 (docs(frontend): adding a SHA1 tutorial with modules)
```
  func.func @add5(%arg0: tensor<32x!FHE.eint<2>>, %arg1: tensor<32x!FHE.eint<2>>, %arg2: tensor<32x!FHE.eint<2>>, %arg3: tensor<32x!FHE.eint<2>>, %arg4: tensor<32x!FHE.eint<2>>) -> tensor<32x!FHE.eint<2>> {
    %0 = "FHE.zero_tensor"() : () -> tensor<32x!FHE.eint<2>>
    %c0 = arith.constant 0 : index
    %extracted = tensor.extract %arg0[%c0] : tensor<32x!FHE.eint<2>>
    %extracted_0 = tensor.extract %arg1[%c0] : tensor<32x!FHE.eint<2>>
    %1 = "FHE.add_eint"(%extracted, %extracted_0) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %c0_i2 = arith.constant 0 : i2
    %2 = "FHE.add_eint_int"(%1, %c0_i2) : (!FHE.eint<2>, i2) -> !FHE.eint<2>
    %c2_i3 = arith.constant 2 : i3
    %cst = arith.constant dense<[0, 0, 1, 1]> : tensor<4xi64>
    %3 = "FHE.apply_lookup_table"(%2, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %cst_1 = arith.constant dense<[0, 1, 0, 1]> : tensor<4xi64>
    %4 = "FHE.apply_lookup_table"(%2, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements = tensor.from_elements %4 : tensor<1x!FHE.eint<2>>
    %inserted_slice = tensor.insert_slice %from_elements into %0[0] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c1 = arith.constant 1 : index
    %extracted_2 = tensor.extract %arg0[%c1] : tensor<32x!FHE.eint<2>>
    %extracted_3 = tensor.extract %arg1[%c1] : tensor<32x!FHE.eint<2>>
    %5 = "FHE.add_eint"(%extracted_2, %extracted_3) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %6 = "FHE.add_eint"(%5, %3) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %7 = "FHE.apply_lookup_table"(%6, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %8 = "FHE.apply_lookup_table"(%6, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_4 = tensor.from_elements %8 : tensor<1x!FHE.eint<2>>
    %inserted_slice_5 = tensor.insert_slice %from_elements_4 into %inserted_slice[1] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c2 = arith.constant 2 : index
    %extracted_6 = tensor.extract %arg0[%c2] : tensor<32x!FHE.eint<2>>
    %extracted_7 = tensor.extract %arg1[%c2] : tensor<32x!FHE.eint<2>>
    %9 = "FHE.add_eint"(%extracted_6, %extracted_7) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %10 = "FHE.add_eint"(%9, %7) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %11 = "FHE.apply_lookup_table"(%10, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %12 = "FHE.apply_lookup_table"(%10, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_8 = tensor.from_elements %12 : tensor<1x!FHE.eint<2>>
    %inserted_slice_9 = tensor.insert_slice %from_elements_8 into %inserted_slice_5[2] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c3 = arith.constant 3 : index
    %extracted_10 = tensor.extract %arg0[%c3] : tensor<32x!FHE.eint<2>>
    %extracted_11 = tensor.extract %arg1[%c3] : tensor<32x!FHE.eint<2>>
    %13 = "FHE.add_eint"(%extracted_10, %extracted_11) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %14 = "FHE.add_eint"(%13, %11) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %15 = "FHE.apply_lookup_table"(%14, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %16 = "FHE.apply_lookup_table"(%14, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_12 = tensor.from_elements %16 : tensor<1x!FHE.eint<2>>
    %inserted_slice_13 = tensor.insert_slice %from_elements_12 into %inserted_slice_9[3] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c4 = arith.constant 4 : index
    %extracted_14 = tensor.extract %arg0[%c4] : tensor<32x!FHE.eint<2>>
    %extracted_15 = tensor.extract %arg1[%c4] : tensor<32x!FHE.eint<2>>
    %17 = "FHE.add_eint"(%extracted_14, %extracted_15) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %18 = "FHE.add_eint"(%17, %15) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %19 = "FHE.apply_lookup_table"(%18, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %20 = "FHE.apply_lookup_table"(%18, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_16 = tensor.from_elements %20 : tensor<1x!FHE.eint<2>>
    %inserted_slice_17 = tensor.insert_slice %from_elements_16 into %inserted_slice_13[4] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c5 = arith.constant 5 : index
    %extracted_18 = tensor.extract %arg0[%c5] : tensor<32x!FHE.eint<2>>
    %extracted_19 = tensor.extract %arg1[%c5] : tensor<32x!FHE.eint<2>>
    %21 = "FHE.add_eint"(%extracted_18, %extracted_19) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %22 = "FHE.add_eint"(%21, %19) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %23 = "FHE.apply_lookup_table"(%22, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %24 = "FHE.apply_lookup_table"(%22, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_20 = tensor.from_elements %24 : tensor<1x!FHE.eint<2>>
    %inserted_slice_21 = tensor.insert_slice %from_elements_20 into %inserted_slice_17[5] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c6 = arith.constant 6 : index
    %extracted_22 = tensor.extract %arg0[%c6] : tensor<32x!FHE.eint<2>>
    %extracted_23 = tensor.extract %arg1[%c6] : tensor<32x!FHE.eint<2>>
    %25 = "FHE.add_eint"(%extracted_22, %extracted_23) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %26 = "FHE.add_eint"(%25, %23) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %27 = "FHE.apply_lookup_table"(%26, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %28 = "FHE.apply_lookup_table"(%26, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_24 = tensor.from_elements %28 : tensor<1x!FHE.eint<2>>
    %inserted_slice_25 = tensor.insert_slice %from_elements_24 into %inserted_slice_21[6] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c7 = arith.constant 7 : index
    %extracted_26 = tensor.extract %arg0[%c7] : tensor<32x!FHE.eint<2>>
    %extracted_27 = tensor.extract %arg1[%c7] : tensor<32x!FHE.eint<2>>
    %29 = "FHE.add_eint"(%extracted_26, %extracted_27) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %30 = "FHE.add_eint"(%29, %27) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %31 = "FHE.apply_lookup_table"(%30, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %32 = "FHE.apply_lookup_table"(%30, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_28 = tensor.from_elements %32 : tensor<1x!FHE.eint<2>>
    %inserted_slice_29 = tensor.insert_slice %from_elements_28 into %inserted_slice_25[7] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c8 = arith.constant 8 : index
    %extracted_30 = tensor.extract %arg0[%c8] : tensor<32x!FHE.eint<2>>
    %extracted_31 = tensor.extract %arg1[%c8] : tensor<32x!FHE.eint<2>>
    %33 = "FHE.add_eint"(%extracted_30, %extracted_31) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %34 = "FHE.add_eint"(%33, %31) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %35 = "FHE.apply_lookup_table"(%34, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %36 = "FHE.apply_lookup_table"(%34, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_32 = tensor.from_elements %36 : tensor<1x!FHE.eint<2>>
    %inserted_slice_33 = tensor.insert_slice %from_elements_32 into %inserted_slice_29[8] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c9 = arith.constant 9 : index
    %extracted_34 = tensor.extract %arg0[%c9] : tensor<32x!FHE.eint<2>>
    %extracted_35 = tensor.extract %arg1[%c9] : tensor<32x!FHE.eint<2>>
    %37 = "FHE.add_eint"(%extracted_34, %extracted_35) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %38 = "FHE.add_eint"(%37, %35) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %39 = "FHE.apply_lookup_table"(%38, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %40 = "FHE.apply_lookup_table"(%38, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_36 = tensor.from_elements %40 : tensor<1x!FHE.eint<2>>
    %inserted_slice_37 = tensor.insert_slice %from_elements_36 into %inserted_slice_33[9] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c10 = arith.constant 10 : index
    %extracted_38 = tensor.extract %arg0[%c10] : tensor<32x!FHE.eint<2>>
    %extracted_39 = tensor.extract %arg1[%c10] : tensor<32x!FHE.eint<2>>
    %41 = "FHE.add_eint"(%extracted_38, %extracted_39) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %42 = "FHE.add_eint"(%41, %39) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %43 = "FHE.apply_lookup_table"(%42, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %44 = "FHE.apply_lookup_table"(%42, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_40 = tensor.from_elements %44 : tensor<1x!FHE.eint<2>>
    %inserted_slice_41 = tensor.insert_slice %from_elements_40 into %inserted_slice_37[10] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c11 = arith.constant 11 : index
    %extracted_42 = tensor.extract %arg0[%c11] : tensor<32x!FHE.eint<2>>
    %extracted_43 = tensor.extract %arg1[%c11] : tensor<32x!FHE.eint<2>>
    %45 = "FHE.add_eint"(%extracted_42, %extracted_43) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %46 = "FHE.add_eint"(%45, %43) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %47 = "FHE.apply_lookup_table"(%46, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %48 = "FHE.apply_lookup_table"(%46, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_44 = tensor.from_elements %48 : tensor<1x!FHE.eint<2>>
    %inserted_slice_45 = tensor.insert_slice %from_elements_44 into %inserted_slice_41[11] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c12 = arith.constant 12 : index
    %extracted_46 = tensor.extract %arg0[%c12] : tensor<32x!FHE.eint<2>>
    %extracted_47 = tensor.extract %arg1[%c12] : tensor<32x!FHE.eint<2>>
    %49 = "FHE.add_eint"(%extracted_46, %extracted_47) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %50 = "FHE.add_eint"(%49, %47) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %51 = "FHE.apply_lookup_table"(%50, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %52 = "FHE.apply_lookup_table"(%50, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_48 = tensor.from_elements %52 : tensor<1x!FHE.eint<2>>
    %inserted_slice_49 = tensor.insert_slice %from_elements_48 into %inserted_slice_45[12] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c13 = arith.constant 13 : index
    %extracted_50 = tensor.extract %arg0[%c13] : tensor<32x!FHE.eint<2>>
    %extracted_51 = tensor.extract %arg1[%c13] : tensor<32x!FHE.eint<2>>
    %53 = "FHE.add_eint"(%extracted_50, %extracted_51) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %54 = "FHE.add_eint"(%53, %51) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %55 = "FHE.apply_lookup_table"(%54, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %56 = "FHE.apply_lookup_table"(%54, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_52 = tensor.from_elements %56 : tensor<1x!FHE.eint<2>>
    %inserted_slice_53 = tensor.insert_slice %from_elements_52 into %inserted_slice_49[13] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c14 = arith.constant 14 : index
    %extracted_54 = tensor.extract %arg0[%c14] : tensor<32x!FHE.eint<2>>
    %extracted_55 = tensor.extract %arg1[%c14] : tensor<32x!FHE.eint<2>>
    %57 = "FHE.add_eint"(%extracted_54, %extracted_55) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %58 = "FHE.add_eint"(%57, %55) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %59 = "FHE.apply_lookup_table"(%58, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %60 = "FHE.apply_lookup_table"(%58, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_56 = tensor.from_elements %60 : tensor<1x!FHE.eint<2>>
    %inserted_slice_57 = tensor.insert_slice %from_elements_56 into %inserted_slice_53[14] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c15 = arith.constant 15 : index
    %extracted_58 = tensor.extract %arg0[%c15] : tensor<32x!FHE.eint<2>>
    %extracted_59 = tensor.extract %arg1[%c15] : tensor<32x!FHE.eint<2>>
    %61 = "FHE.add_eint"(%extracted_58, %extracted_59) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %62 = "FHE.add_eint"(%61, %59) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %63 = "FHE.apply_lookup_table"(%62, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %64 = "FHE.apply_lookup_table"(%62, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_60 = tensor.from_elements %64 : tensor<1x!FHE.eint<2>>
    %inserted_slice_61 = tensor.insert_slice %from_elements_60 into %inserted_slice_57[15] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c16 = arith.constant 16 : index
    %extracted_62 = tensor.extract %arg0[%c16] : tensor<32x!FHE.eint<2>>
    %extracted_63 = tensor.extract %arg1[%c16] : tensor<32x!FHE.eint<2>>
    %65 = "FHE.add_eint"(%extracted_62, %extracted_63) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %66 = "FHE.add_eint"(%65, %63) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %67 = "FHE.apply_lookup_table"(%66, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %68 = "FHE.apply_lookup_table"(%66, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_64 = tensor.from_elements %68 : tensor<1x!FHE.eint<2>>
    %inserted_slice_65 = tensor.insert_slice %from_elements_64 into %inserted_slice_61[16] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c17 = arith.constant 17 : index
    %extracted_66 = tensor.extract %arg0[%c17] : tensor<32x!FHE.eint<2>>
    %extracted_67 = tensor.extract %arg1[%c17] : tensor<32x!FHE.eint<2>>
    %69 = "FHE.add_eint"(%extracted_66, %extracted_67) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %70 = "FHE.add_eint"(%69, %67) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %71 = "FHE.apply_lookup_table"(%70, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %72 = "FHE.apply_lookup_table"(%70, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_68 = tensor.from_elements %72 : tensor<1x!FHE.eint<2>>
    %inserted_slice_69 = tensor.insert_slice %from_elements_68 into %inserted_slice_65[17] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c18 = arith.constant 18 : index
    %extracted_70 = tensor.extract %arg0[%c18] : tensor<32x!FHE.eint<2>>
    %extracted_71 = tensor.extract %arg1[%c18] : tensor<32x!FHE.eint<2>>
    %73 = "FHE.add_eint"(%extracted_70, %extracted_71) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %74 = "FHE.add_eint"(%73, %71) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %75 = "FHE.apply_lookup_table"(%74, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %76 = "FHE.apply_lookup_table"(%74, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_72 = tensor.from_elements %76 : tensor<1x!FHE.eint<2>>
    %inserted_slice_73 = tensor.insert_slice %from_elements_72 into %inserted_slice_69[18] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c19 = arith.constant 19 : index
    %extracted_74 = tensor.extract %arg0[%c19] : tensor<32x!FHE.eint<2>>
    %extracted_75 = tensor.extract %arg1[%c19] : tensor<32x!FHE.eint<2>>
    %77 = "FHE.add_eint"(%extracted_74, %extracted_75) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %78 = "FHE.add_eint"(%77, %75) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %79 = "FHE.apply_lookup_table"(%78, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %80 = "FHE.apply_lookup_table"(%78, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_76 = tensor.from_elements %80 : tensor<1x!FHE.eint<2>>
    %inserted_slice_77 = tensor.insert_slice %from_elements_76 into %inserted_slice_73[19] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c20 = arith.constant 20 : index
    %extracted_78 = tensor.extract %arg0[%c20] : tensor<32x!FHE.eint<2>>
    %extracted_79 = tensor.extract %arg1[%c20] : tensor<32x!FHE.eint<2>>
    %81 = "FHE.add_eint"(%extracted_78, %extracted_79) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %82 = "FHE.add_eint"(%81, %79) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %83 = "FHE.apply_lookup_table"(%82, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %84 = "FHE.apply_lookup_table"(%82, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_80 = tensor.from_elements %84 : tensor<1x!FHE.eint<2>>
    %inserted_slice_81 = tensor.insert_slice %from_elements_80 into %inserted_slice_77[20] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c21 = arith.constant 21 : index
    %extracted_82 = tensor.extract %arg0[%c21] : tensor<32x!FHE.eint<2>>
    %extracted_83 = tensor.extract %arg1[%c21] : tensor<32x!FHE.eint<2>>
    %85 = "FHE.add_eint"(%extracted_82, %extracted_83) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %86 = "FHE.add_eint"(%85, %83) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %87 = "FHE.apply_lookup_table"(%86, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %88 = "FHE.apply_lookup_table"(%86, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_84 = tensor.from_elements %88 : tensor<1x!FHE.eint<2>>
    %inserted_slice_85 = tensor.insert_slice %from_elements_84 into %inserted_slice_81[21] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c22 = arith.constant 22 : index
    %extracted_86 = tensor.extract %arg0[%c22] : tensor<32x!FHE.eint<2>>
    %extracted_87 = tensor.extract %arg1[%c22] : tensor<32x!FHE.eint<2>>
    %89 = "FHE.add_eint"(%extracted_86, %extracted_87) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %90 = "FHE.add_eint"(%89, %87) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %91 = "FHE.apply_lookup_table"(%90, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %92 = "FHE.apply_lookup_table"(%90, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_88 = tensor.from_elements %92 : tensor<1x!FHE.eint<2>>
    %inserted_slice_89 = tensor.insert_slice %from_elements_88 into %inserted_slice_85[22] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c23 = arith.constant 23 : index
    %extracted_90 = tensor.extract %arg0[%c23] : tensor<32x!FHE.eint<2>>
    %extracted_91 = tensor.extract %arg1[%c23] : tensor<32x!FHE.eint<2>>
    %93 = "FHE.add_eint"(%extracted_90, %extracted_91) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %94 = "FHE.add_eint"(%93, %91) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %95 = "FHE.apply_lookup_table"(%94, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %96 = "FHE.apply_lookup_table"(%94, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_92 = tensor.from_elements %96 : tensor<1x!FHE.eint<2>>
    %inserted_slice_93 = tensor.insert_slice %from_elements_92 into %inserted_slice_89[23] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c24 = arith.constant 24 : index
    %extracted_94 = tensor.extract %arg0[%c24] : tensor<32x!FHE.eint<2>>
    %extracted_95 = tensor.extract %arg1[%c24] : tensor<32x!FHE.eint<2>>
    %97 = "FHE.add_eint"(%extracted_94, %extracted_95) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %98 = "FHE.add_eint"(%97, %95) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %99 = "FHE.apply_lookup_table"(%98, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %100 = "FHE.apply_lookup_table"(%98, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_96 = tensor.from_elements %100 : tensor<1x!FHE.eint<2>>
    %inserted_slice_97 = tensor.insert_slice %from_elements_96 into %inserted_slice_93[24] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c25 = arith.constant 25 : index
    %extracted_98 = tensor.extract %arg0[%c25] : tensor<32x!FHE.eint<2>>
    %extracted_99 = tensor.extract %arg1[%c25] : tensor<32x!FHE.eint<2>>
    %101 = "FHE.add_eint"(%extracted_98, %extracted_99) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %102 = "FHE.add_eint"(%101, %99) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %103 = "FHE.apply_lookup_table"(%102, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %104 = "FHE.apply_lookup_table"(%102, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_100 = tensor.from_elements %104 : tensor<1x!FHE.eint<2>>
    %inserted_slice_101 = tensor.insert_slice %from_elements_100 into %inserted_slice_97[25] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c26 = arith.constant 26 : index
    %extracted_102 = tensor.extract %arg0[%c26] : tensor<32x!FHE.eint<2>>
    %extracted_103 = tensor.extract %arg1[%c26] : tensor<32x!FHE.eint<2>>
    %105 = "FHE.add_eint"(%extracted_102, %extracted_103) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %106 = "FHE.add_eint"(%105, %103) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %107 = "FHE.apply_lookup_table"(%106, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %108 = "FHE.apply_lookup_table"(%106, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_104 = tensor.from_elements %108 : tensor<1x!FHE.eint<2>>
    %inserted_slice_105 = tensor.insert_slice %from_elements_104 into %inserted_slice_101[26] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c27 = arith.constant 27 : index
    %extracted_106 = tensor.extract %arg0[%c27] : tensor<32x!FHE.eint<2>>
    %extracted_107 = tensor.extract %arg1[%c27] : tensor<32x!FHE.eint<2>>
    %109 = "FHE.add_eint"(%extracted_106, %extracted_107) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %110 = "FHE.add_eint"(%109, %107) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %111 = "FHE.apply_lookup_table"(%110, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %112 = "FHE.apply_lookup_table"(%110, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_108 = tensor.from_elements %112 : tensor<1x!FHE.eint<2>>
    %inserted_slice_109 = tensor.insert_slice %from_elements_108 into %inserted_slice_105[27] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c28 = arith.constant 28 : index
    %extracted_110 = tensor.extract %arg0[%c28] : tensor<32x!FHE.eint<2>>
    %extracted_111 = tensor.extract %arg1[%c28] : tensor<32x!FHE.eint<2>>
    %113 = "FHE.add_eint"(%extracted_110, %extracted_111) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %114 = "FHE.add_eint"(%113, %111) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %115 = "FHE.apply_lookup_table"(%114, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %116 = "FHE.apply_lookup_table"(%114, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_112 = tensor.from_elements %116 : tensor<1x!FHE.eint<2>>
    %inserted_slice_113 = tensor.insert_slice %from_elements_112 into %inserted_slice_109[28] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c29 = arith.constant 29 : index
    %extracted_114 = tensor.extract %arg0[%c29] : tensor<32x!FHE.eint<2>>
    %extracted_115 = tensor.extract %arg1[%c29] : tensor<32x!FHE.eint<2>>
    %117 = "FHE.add_eint"(%extracted_114, %extracted_115) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %118 = "FHE.add_eint"(%117, %115) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %119 = "FHE.apply_lookup_table"(%118, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %120 = "FHE.apply_lookup_table"(%118, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_116 = tensor.from_elements %120 : tensor<1x!FHE.eint<2>>
    %inserted_slice_117 = tensor.insert_slice %from_elements_116 into %inserted_slice_113[29] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c30 = arith.constant 30 : index
    %extracted_118 = tensor.extract %arg0[%c30] : tensor<32x!FHE.eint<2>>
    %extracted_119 = tensor.extract %arg1[%c30] : tensor<32x!FHE.eint<2>>
    %121 = "FHE.add_eint"(%extracted_118, %extracted_119) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %122 = "FHE.add_eint"(%121, %119) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %123 = "FHE.apply_lookup_table"(%122, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %124 = "FHE.apply_lookup_table"(%122, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_120 = tensor.from_elements %124 : tensor<1x!FHE.eint<2>>
    %inserted_slice_121 = tensor.insert_slice %from_elements_120 into %inserted_slice_117[30] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %c31 = arith.constant 31 : index
    %extracted_122 = tensor.extract %arg0[%c31] : tensor<32x!FHE.eint<2>>
    %extracted_123 = tensor.extract %arg1[%c31] : tensor<32x!FHE.eint<2>>
    %125 = "FHE.add_eint"(%extracted_122, %extracted_123) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %126 = "FHE.add_eint"(%125, %123) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %127 = "FHE.apply_lookup_table"(%126, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_124 = tensor.from_elements %127 : tensor<1x!FHE.eint<2>>
    %inserted_slice_125 = tensor.insert_slice %from_elements_124 into %inserted_slice_121[31] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_126 = tensor.extract %inserted_slice_125[%c0] : tensor<32x!FHE.eint<2>>
    %extracted_127 = tensor.extract %arg2[%c0] : tensor<32x!FHE.eint<2>>
    %128 = "FHE.add_eint"(%extracted_126, %extracted_127) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %129 = "FHE.add_eint_int"(%128, %c0_i2) : (!FHE.eint<2>, i2) -> !FHE.eint<2>
    %130 = "FHE.apply_lookup_table"(%129, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %131 = "FHE.apply_lookup_table"(%129, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_128 = tensor.from_elements %131 : tensor<1x!FHE.eint<2>>
    %inserted_slice_129 = tensor.insert_slice %from_elements_128 into %inserted_slice_125[0] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_130 = tensor.extract %inserted_slice_129[%c1] : tensor<32x!FHE.eint<2>>
    %extracted_131 = tensor.extract %arg2[%c1] : tensor<32x!FHE.eint<2>>
    %132 = "FHE.add_eint"(%extracted_130, %extracted_131) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %133 = "FHE.add_eint"(%132, %130) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %134 = "FHE.apply_lookup_table"(%133, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %135 = "FHE.apply_lookup_table"(%133, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_132 = tensor.from_elements %135 : tensor<1x!FHE.eint<2>>
    %inserted_slice_133 = tensor.insert_slice %from_elements_132 into %inserted_slice_129[1] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_134 = tensor.extract %inserted_slice_133[%c2] : tensor<32x!FHE.eint<2>>
    %extracted_135 = tensor.extract %arg2[%c2] : tensor<32x!FHE.eint<2>>
    %136 = "FHE.add_eint"(%extracted_134, %extracted_135) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %137 = "FHE.add_eint"(%136, %134) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %138 = "FHE.apply_lookup_table"(%137, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %139 = "FHE.apply_lookup_table"(%137, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_136 = tensor.from_elements %139 : tensor<1x!FHE.eint<2>>
    %inserted_slice_137 = tensor.insert_slice %from_elements_136 into %inserted_slice_133[2] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_138 = tensor.extract %inserted_slice_137[%c3] : tensor<32x!FHE.eint<2>>
    %extracted_139 = tensor.extract %arg2[%c3] : tensor<32x!FHE.eint<2>>
    %140 = "FHE.add_eint"(%extracted_138, %extracted_139) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %141 = "FHE.add_eint"(%140, %138) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %142 = "FHE.apply_lookup_table"(%141, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %143 = "FHE.apply_lookup_table"(%141, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_140 = tensor.from_elements %143 : tensor<1x!FHE.eint<2>>
    %inserted_slice_141 = tensor.insert_slice %from_elements_140 into %inserted_slice_137[3] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_142 = tensor.extract %inserted_slice_141[%c4] : tensor<32x!FHE.eint<2>>
    %extracted_143 = tensor.extract %arg2[%c4] : tensor<32x!FHE.eint<2>>
    %144 = "FHE.add_eint"(%extracted_142, %extracted_143) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %145 = "FHE.add_eint"(%144, %142) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %146 = "FHE.apply_lookup_table"(%145, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %147 = "FHE.apply_lookup_table"(%145, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_144 = tensor.from_elements %147 : tensor<1x!FHE.eint<2>>
    %inserted_slice_145 = tensor.insert_slice %from_elements_144 into %inserted_slice_141[4] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_146 = tensor.extract %inserted_slice_145[%c5] : tensor<32x!FHE.eint<2>>
    %extracted_147 = tensor.extract %arg2[%c5] : tensor<32x!FHE.eint<2>>
    %148 = "FHE.add_eint"(%extracted_146, %extracted_147) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %149 = "FHE.add_eint"(%148, %146) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %150 = "FHE.apply_lookup_table"(%149, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %151 = "FHE.apply_lookup_table"(%149, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_148 = tensor.from_elements %151 : tensor<1x!FHE.eint<2>>
    %inserted_slice_149 = tensor.insert_slice %from_elements_148 into %inserted_slice_145[5] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_150 = tensor.extract %inserted_slice_149[%c6] : tensor<32x!FHE.eint<2>>
    %extracted_151 = tensor.extract %arg2[%c6] : tensor<32x!FHE.eint<2>>
    %152 = "FHE.add_eint"(%extracted_150, %extracted_151) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %153 = "FHE.add_eint"(%152, %150) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %154 = "FHE.apply_lookup_table"(%153, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %155 = "FHE.apply_lookup_table"(%153, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_152 = tensor.from_elements %155 : tensor<1x!FHE.eint<2>>
    %inserted_slice_153 = tensor.insert_slice %from_elements_152 into %inserted_slice_149[6] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_154 = tensor.extract %inserted_slice_153[%c7] : tensor<32x!FHE.eint<2>>
    %extracted_155 = tensor.extract %arg2[%c7] : tensor<32x!FHE.eint<2>>
    %156 = "FHE.add_eint"(%extracted_154, %extracted_155) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %157 = "FHE.add_eint"(%156, %154) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %158 = "FHE.apply_lookup_table"(%157, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %159 = "FHE.apply_lookup_table"(%157, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_156 = tensor.from_elements %159 : tensor<1x!FHE.eint<2>>
    %inserted_slice_157 = tensor.insert_slice %from_elements_156 into %inserted_slice_153[7] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_158 = tensor.extract %inserted_slice_157[%c8] : tensor<32x!FHE.eint<2>>
    %extracted_159 = tensor.extract %arg2[%c8] : tensor<32x!FHE.eint<2>>
    %160 = "FHE.add_eint"(%extracted_158, %extracted_159) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %161 = "FHE.add_eint"(%160, %158) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %162 = "FHE.apply_lookup_table"(%161, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %163 = "FHE.apply_lookup_table"(%161, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_160 = tensor.from_elements %163 : tensor<1x!FHE.eint<2>>
    %inserted_slice_161 = tensor.insert_slice %from_elements_160 into %inserted_slice_157[8] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_162 = tensor.extract %inserted_slice_161[%c9] : tensor<32x!FHE.eint<2>>
    %extracted_163 = tensor.extract %arg2[%c9] : tensor<32x!FHE.eint<2>>
    %164 = "FHE.add_eint"(%extracted_162, %extracted_163) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %165 = "FHE.add_eint"(%164, %162) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %166 = "FHE.apply_lookup_table"(%165, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %167 = "FHE.apply_lookup_table"(%165, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_164 = tensor.from_elements %167 : tensor<1x!FHE.eint<2>>
    %inserted_slice_165 = tensor.insert_slice %from_elements_164 into %inserted_slice_161[9] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_166 = tensor.extract %inserted_slice_165[%c10] : tensor<32x!FHE.eint<2>>
    %extracted_167 = tensor.extract %arg2[%c10] : tensor<32x!FHE.eint<2>>
    %168 = "FHE.add_eint"(%extracted_166, %extracted_167) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %169 = "FHE.add_eint"(%168, %166) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %170 = "FHE.apply_lookup_table"(%169, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %171 = "FHE.apply_lookup_table"(%169, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_168 = tensor.from_elements %171 : tensor<1x!FHE.eint<2>>
    %inserted_slice_169 = tensor.insert_slice %from_elements_168 into %inserted_slice_165[10] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_170 = tensor.extract %inserted_slice_169[%c11] : tensor<32x!FHE.eint<2>>
    %extracted_171 = tensor.extract %arg2[%c11] : tensor<32x!FHE.eint<2>>
    %172 = "FHE.add_eint"(%extracted_170, %extracted_171) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %173 = "FHE.add_eint"(%172, %170) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %174 = "FHE.apply_lookup_table"(%173, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %175 = "FHE.apply_lookup_table"(%173, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_172 = tensor.from_elements %175 : tensor<1x!FHE.eint<2>>
    %inserted_slice_173 = tensor.insert_slice %from_elements_172 into %inserted_slice_169[11] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_174 = tensor.extract %inserted_slice_173[%c12] : tensor<32x!FHE.eint<2>>
    %extracted_175 = tensor.extract %arg2[%c12] : tensor<32x!FHE.eint<2>>
    %176 = "FHE.add_eint"(%extracted_174, %extracted_175) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %177 = "FHE.add_eint"(%176, %174) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %178 = "FHE.apply_lookup_table"(%177, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %179 = "FHE.apply_lookup_table"(%177, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_176 = tensor.from_elements %179 : tensor<1x!FHE.eint<2>>
    %inserted_slice_177 = tensor.insert_slice %from_elements_176 into %inserted_slice_173[12] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_178 = tensor.extract %inserted_slice_177[%c13] : tensor<32x!FHE.eint<2>>
    %extracted_179 = tensor.extract %arg2[%c13] : tensor<32x!FHE.eint<2>>
    %180 = "FHE.add_eint"(%extracted_178, %extracted_179) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %181 = "FHE.add_eint"(%180, %178) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %182 = "FHE.apply_lookup_table"(%181, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %183 = "FHE.apply_lookup_table"(%181, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_180 = tensor.from_elements %183 : tensor<1x!FHE.eint<2>>
    %inserted_slice_181 = tensor.insert_slice %from_elements_180 into %inserted_slice_177[13] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_182 = tensor.extract %inserted_slice_181[%c14] : tensor<32x!FHE.eint<2>>
    %extracted_183 = tensor.extract %arg2[%c14] : tensor<32x!FHE.eint<2>>
    %184 = "FHE.add_eint"(%extracted_182, %extracted_183) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %185 = "FHE.add_eint"(%184, %182) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %186 = "FHE.apply_lookup_table"(%185, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %187 = "FHE.apply_lookup_table"(%185, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_184 = tensor.from_elements %187 : tensor<1x!FHE.eint<2>>
    %inserted_slice_185 = tensor.insert_slice %from_elements_184 into %inserted_slice_181[14] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_186 = tensor.extract %inserted_slice_185[%c15] : tensor<32x!FHE.eint<2>>
    %extracted_187 = tensor.extract %arg2[%c15] : tensor<32x!FHE.eint<2>>
    %188 = "FHE.add_eint"(%extracted_186, %extracted_187) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %189 = "FHE.add_eint"(%188, %186) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %190 = "FHE.apply_lookup_table"(%189, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %191 = "FHE.apply_lookup_table"(%189, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_188 = tensor.from_elements %191 : tensor<1x!FHE.eint<2>>
    %inserted_slice_189 = tensor.insert_slice %from_elements_188 into %inserted_slice_185[15] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_190 = tensor.extract %inserted_slice_189[%c16] : tensor<32x!FHE.eint<2>>
    %extracted_191 = tensor.extract %arg2[%c16] : tensor<32x!FHE.eint<2>>
    %192 = "FHE.add_eint"(%extracted_190, %extracted_191) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %193 = "FHE.add_eint"(%192, %190) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %194 = "FHE.apply_lookup_table"(%193, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %195 = "FHE.apply_lookup_table"(%193, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_192 = tensor.from_elements %195 : tensor<1x!FHE.eint<2>>
    %inserted_slice_193 = tensor.insert_slice %from_elements_192 into %inserted_slice_189[16] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_194 = tensor.extract %inserted_slice_193[%c17] : tensor<32x!FHE.eint<2>>
    %extracted_195 = tensor.extract %arg2[%c17] : tensor<32x!FHE.eint<2>>
    %196 = "FHE.add_eint"(%extracted_194, %extracted_195) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %197 = "FHE.add_eint"(%196, %194) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %198 = "FHE.apply_lookup_table"(%197, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %199 = "FHE.apply_lookup_table"(%197, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_196 = tensor.from_elements %199 : tensor<1x!FHE.eint<2>>
    %inserted_slice_197 = tensor.insert_slice %from_elements_196 into %inserted_slice_193[17] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_198 = tensor.extract %inserted_slice_197[%c18] : tensor<32x!FHE.eint<2>>
    %extracted_199 = tensor.extract %arg2[%c18] : tensor<32x!FHE.eint<2>>
    %200 = "FHE.add_eint"(%extracted_198, %extracted_199) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %201 = "FHE.add_eint"(%200, %198) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %202 = "FHE.apply_lookup_table"(%201, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %203 = "FHE.apply_lookup_table"(%201, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_200 = tensor.from_elements %203 : tensor<1x!FHE.eint<2>>
    %inserted_slice_201 = tensor.insert_slice %from_elements_200 into %inserted_slice_197[18] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_202 = tensor.extract %inserted_slice_201[%c19] : tensor<32x!FHE.eint<2>>
    %extracted_203 = tensor.extract %arg2[%c19] : tensor<32x!FHE.eint<2>>
    %204 = "FHE.add_eint"(%extracted_202, %extracted_203) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %205 = "FHE.add_eint"(%204, %202) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %206 = "FHE.apply_lookup_table"(%205, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %207 = "FHE.apply_lookup_table"(%205, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_204 = tensor.from_elements %207 : tensor<1x!FHE.eint<2>>
    %inserted_slice_205 = tensor.insert_slice %from_elements_204 into %inserted_slice_201[19] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_206 = tensor.extract %inserted_slice_205[%c20] : tensor<32x!FHE.eint<2>>
    %extracted_207 = tensor.extract %arg2[%c20] : tensor<32x!FHE.eint<2>>
    %208 = "FHE.add_eint"(%extracted_206, %extracted_207) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %209 = "FHE.add_eint"(%208, %206) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %210 = "FHE.apply_lookup_table"(%209, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %211 = "FHE.apply_lookup_table"(%209, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_208 = tensor.from_elements %211 : tensor<1x!FHE.eint<2>>
    %inserted_slice_209 = tensor.insert_slice %from_elements_208 into %inserted_slice_205[20] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_210 = tensor.extract %inserted_slice_209[%c21] : tensor<32x!FHE.eint<2>>
    %extracted_211 = tensor.extract %arg2[%c21] : tensor<32x!FHE.eint<2>>
    %212 = "FHE.add_eint"(%extracted_210, %extracted_211) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %213 = "FHE.add_eint"(%212, %210) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %214 = "FHE.apply_lookup_table"(%213, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %215 = "FHE.apply_lookup_table"(%213, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_212 = tensor.from_elements %215 : tensor<1x!FHE.eint<2>>
    %inserted_slice_213 = tensor.insert_slice %from_elements_212 into %inserted_slice_209[21] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_214 = tensor.extract %inserted_slice_213[%c22] : tensor<32x!FHE.eint<2>>
    %extracted_215 = tensor.extract %arg2[%c22] : tensor<32x!FHE.eint<2>>
    %216 = "FHE.add_eint"(%extracted_214, %extracted_215) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %217 = "FHE.add_eint"(%216, %214) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %218 = "FHE.apply_lookup_table"(%217, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %219 = "FHE.apply_lookup_table"(%217, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_216 = tensor.from_elements %219 : tensor<1x!FHE.eint<2>>
    %inserted_slice_217 = tensor.insert_slice %from_elements_216 into %inserted_slice_213[22] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_218 = tensor.extract %inserted_slice_217[%c23] : tensor<32x!FHE.eint<2>>
    %extracted_219 = tensor.extract %arg2[%c23] : tensor<32x!FHE.eint<2>>
    %220 = "FHE.add_eint"(%extracted_218, %extracted_219) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %221 = "FHE.add_eint"(%220, %218) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %222 = "FHE.apply_lookup_table"(%221, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %223 = "FHE.apply_lookup_table"(%221, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_220 = tensor.from_elements %223 : tensor<1x!FHE.eint<2>>
    %inserted_slice_221 = tensor.insert_slice %from_elements_220 into %inserted_slice_217[23] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_222 = tensor.extract %inserted_slice_221[%c24] : tensor<32x!FHE.eint<2>>
    %extracted_223 = tensor.extract %arg2[%c24] : tensor<32x!FHE.eint<2>>
    %224 = "FHE.add_eint"(%extracted_222, %extracted_223) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %225 = "FHE.add_eint"(%224, %222) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %226 = "FHE.apply_lookup_table"(%225, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %227 = "FHE.apply_lookup_table"(%225, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_224 = tensor.from_elements %227 : tensor<1x!FHE.eint<2>>
    %inserted_slice_225 = tensor.insert_slice %from_elements_224 into %inserted_slice_221[24] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_226 = tensor.extract %inserted_slice_225[%c25] : tensor<32x!FHE.eint<2>>
    %extracted_227 = tensor.extract %arg2[%c25] : tensor<32x!FHE.eint<2>>
    %228 = "FHE.add_eint"(%extracted_226, %extracted_227) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %229 = "FHE.add_eint"(%228, %226) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %230 = "FHE.apply_lookup_table"(%229, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %231 = "FHE.apply_lookup_table"(%229, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_228 = tensor.from_elements %231 : tensor<1x!FHE.eint<2>>
    %inserted_slice_229 = tensor.insert_slice %from_elements_228 into %inserted_slice_225[25] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_230 = tensor.extract %inserted_slice_229[%c26] : tensor<32x!FHE.eint<2>>
    %extracted_231 = tensor.extract %arg2[%c26] : tensor<32x!FHE.eint<2>>
    %232 = "FHE.add_eint"(%extracted_230, %extracted_231) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %233 = "FHE.add_eint"(%232, %230) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %234 = "FHE.apply_lookup_table"(%233, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %235 = "FHE.apply_lookup_table"(%233, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_232 = tensor.from_elements %235 : tensor<1x!FHE.eint<2>>
    %inserted_slice_233 = tensor.insert_slice %from_elements_232 into %inserted_slice_229[26] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_234 = tensor.extract %inserted_slice_233[%c27] : tensor<32x!FHE.eint<2>>
    %extracted_235 = tensor.extract %arg2[%c27] : tensor<32x!FHE.eint<2>>
    %236 = "FHE.add_eint"(%extracted_234, %extracted_235) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %237 = "FHE.add_eint"(%236, %234) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %238 = "FHE.apply_lookup_table"(%237, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %239 = "FHE.apply_lookup_table"(%237, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_236 = tensor.from_elements %239 : tensor<1x!FHE.eint<2>>
    %inserted_slice_237 = tensor.insert_slice %from_elements_236 into %inserted_slice_233[27] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_238 = tensor.extract %inserted_slice_237[%c28] : tensor<32x!FHE.eint<2>>
    %extracted_239 = tensor.extract %arg2[%c28] : tensor<32x!FHE.eint<2>>
    %240 = "FHE.add_eint"(%extracted_238, %extracted_239) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %241 = "FHE.add_eint"(%240, %238) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %242 = "FHE.apply_lookup_table"(%241, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %243 = "FHE.apply_lookup_table"(%241, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_240 = tensor.from_elements %243 : tensor<1x!FHE.eint<2>>
    %inserted_slice_241 = tensor.insert_slice %from_elements_240 into %inserted_slice_237[28] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_242 = tensor.extract %inserted_slice_241[%c29] : tensor<32x!FHE.eint<2>>
    %extracted_243 = tensor.extract %arg2[%c29] : tensor<32x!FHE.eint<2>>
    %244 = "FHE.add_eint"(%extracted_242, %extracted_243) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %245 = "FHE.add_eint"(%244, %242) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %246 = "FHE.apply_lookup_table"(%245, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %247 = "FHE.apply_lookup_table"(%245, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_244 = tensor.from_elements %247 : tensor<1x!FHE.eint<2>>
    %inserted_slice_245 = tensor.insert_slice %from_elements_244 into %inserted_slice_241[29] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_246 = tensor.extract %inserted_slice_245[%c30] : tensor<32x!FHE.eint<2>>
    %extracted_247 = tensor.extract %arg2[%c30] : tensor<32x!FHE.eint<2>>
    %248 = "FHE.add_eint"(%extracted_246, %extracted_247) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %249 = "FHE.add_eint"(%248, %246) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %250 = "FHE.apply_lookup_table"(%249, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %251 = "FHE.apply_lookup_table"(%249, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_248 = tensor.from_elements %251 : tensor<1x!FHE.eint<2>>
    %inserted_slice_249 = tensor.insert_slice %from_elements_248 into %inserted_slice_245[30] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_250 = tensor.extract %inserted_slice_249[%c31] : tensor<32x!FHE.eint<2>>
    %extracted_251 = tensor.extract %arg2[%c31] : tensor<32x!FHE.eint<2>>
    %252 = "FHE.add_eint"(%extracted_250, %extracted_251) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %253 = "FHE.add_eint"(%252, %250) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %254 = "FHE.apply_lookup_table"(%253, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_252 = tensor.from_elements %254 : tensor<1x!FHE.eint<2>>
    %inserted_slice_253 = tensor.insert_slice %from_elements_252 into %inserted_slice_249[31] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_254 = tensor.extract %inserted_slice_253[%c0] : tensor<32x!FHE.eint<2>>
    %extracted_255 = tensor.extract %arg3[%c0] : tensor<32x!FHE.eint<2>>
    %255 = "FHE.add_eint"(%extracted_254, %extracted_255) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %256 = "FHE.add_eint_int"(%255, %c0_i2) : (!FHE.eint<2>, i2) -> !FHE.eint<2>
    %257 = "FHE.apply_lookup_table"(%256, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %258 = "FHE.apply_lookup_table"(%256, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_256 = tensor.from_elements %258 : tensor<1x!FHE.eint<2>>
    %inserted_slice_257 = tensor.insert_slice %from_elements_256 into %inserted_slice_253[0] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_258 = tensor.extract %inserted_slice_257[%c1] : tensor<32x!FHE.eint<2>>
    %extracted_259 = tensor.extract %arg3[%c1] : tensor<32x!FHE.eint<2>>
    %259 = "FHE.add_eint"(%extracted_258, %extracted_259) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %260 = "FHE.add_eint"(%259, %257) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %261 = "FHE.apply_lookup_table"(%260, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %262 = "FHE.apply_lookup_table"(%260, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_260 = tensor.from_elements %262 : tensor<1x!FHE.eint<2>>
    %inserted_slice_261 = tensor.insert_slice %from_elements_260 into %inserted_slice_257[1] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_262 = tensor.extract %inserted_slice_261[%c2] : tensor<32x!FHE.eint<2>>
    %extracted_263 = tensor.extract %arg3[%c2] : tensor<32x!FHE.eint<2>>
    %263 = "FHE.add_eint"(%extracted_262, %extracted_263) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %264 = "FHE.add_eint"(%263, %261) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %265 = "FHE.apply_lookup_table"(%264, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %266 = "FHE.apply_lookup_table"(%264, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_264 = tensor.from_elements %266 : tensor<1x!FHE.eint<2>>
    %inserted_slice_265 = tensor.insert_slice %from_elements_264 into %inserted_slice_261[2] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_266 = tensor.extract %inserted_slice_265[%c3] : tensor<32x!FHE.eint<2>>
    %extracted_267 = tensor.extract %arg3[%c3] : tensor<32x!FHE.eint<2>>
    %267 = "FHE.add_eint"(%extracted_266, %extracted_267) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %268 = "FHE.add_eint"(%267, %265) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %269 = "FHE.apply_lookup_table"(%268, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %270 = "FHE.apply_lookup_table"(%268, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_268 = tensor.from_elements %270 : tensor<1x!FHE.eint<2>>
    %inserted_slice_269 = tensor.insert_slice %from_elements_268 into %inserted_slice_265[3] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_270 = tensor.extract %inserted_slice_269[%c4] : tensor<32x!FHE.eint<2>>
    %extracted_271 = tensor.extract %arg3[%c4] : tensor<32x!FHE.eint<2>>
    %271 = "FHE.add_eint"(%extracted_270, %extracted_271) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %272 = "FHE.add_eint"(%271, %269) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %273 = "FHE.apply_lookup_table"(%272, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %274 = "FHE.apply_lookup_table"(%272, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_272 = tensor.from_elements %274 : tensor<1x!FHE.eint<2>>
    %inserted_slice_273 = tensor.insert_slice %from_elements_272 into %inserted_slice_269[4] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_274 = tensor.extract %inserted_slice_273[%c5] : tensor<32x!FHE.eint<2>>
    %extracted_275 = tensor.extract %arg3[%c5] : tensor<32x!FHE.eint<2>>
    %275 = "FHE.add_eint"(%extracted_274, %extracted_275) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %276 = "FHE.add_eint"(%275, %273) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %277 = "FHE.apply_lookup_table"(%276, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %278 = "FHE.apply_lookup_table"(%276, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_276 = tensor.from_elements %278 : tensor<1x!FHE.eint<2>>
    %inserted_slice_277 = tensor.insert_slice %from_elements_276 into %inserted_slice_273[5] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_278 = tensor.extract %inserted_slice_277[%c6] : tensor<32x!FHE.eint<2>>
    %extracted_279 = tensor.extract %arg3[%c6] : tensor<32x!FHE.eint<2>>
    %279 = "FHE.add_eint"(%extracted_278, %extracted_279) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %280 = "FHE.add_eint"(%279, %277) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %281 = "FHE.apply_lookup_table"(%280, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %282 = "FHE.apply_lookup_table"(%280, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_280 = tensor.from_elements %282 : tensor<1x!FHE.eint<2>>
    %inserted_slice_281 = tensor.insert_slice %from_elements_280 into %inserted_slice_277[6] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_282 = tensor.extract %inserted_slice_281[%c7] : tensor<32x!FHE.eint<2>>
    %extracted_283 = tensor.extract %arg3[%c7] : tensor<32x!FHE.eint<2>>
    %283 = "FHE.add_eint"(%extracted_282, %extracted_283) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %284 = "FHE.add_eint"(%283, %281) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %285 = "FHE.apply_lookup_table"(%284, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %286 = "FHE.apply_lookup_table"(%284, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_284 = tensor.from_elements %286 : tensor<1x!FHE.eint<2>>
    %inserted_slice_285 = tensor.insert_slice %from_elements_284 into %inserted_slice_281[7] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_286 = tensor.extract %inserted_slice_285[%c8] : tensor<32x!FHE.eint<2>>
    %extracted_287 = tensor.extract %arg3[%c8] : tensor<32x!FHE.eint<2>>
    %287 = "FHE.add_eint"(%extracted_286, %extracted_287) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %288 = "FHE.add_eint"(%287, %285) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %289 = "FHE.apply_lookup_table"(%288, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %290 = "FHE.apply_lookup_table"(%288, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_288 = tensor.from_elements %290 : tensor<1x!FHE.eint<2>>
    %inserted_slice_289 = tensor.insert_slice %from_elements_288 into %inserted_slice_285[8] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_290 = tensor.extract %inserted_slice_289[%c9] : tensor<32x!FHE.eint<2>>
    %extracted_291 = tensor.extract %arg3[%c9] : tensor<32x!FHE.eint<2>>
    %291 = "FHE.add_eint"(%extracted_290, %extracted_291) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %292 = "FHE.add_eint"(%291, %289) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %293 = "FHE.apply_lookup_table"(%292, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %294 = "FHE.apply_lookup_table"(%292, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_292 = tensor.from_elements %294 : tensor<1x!FHE.eint<2>>
    %inserted_slice_293 = tensor.insert_slice %from_elements_292 into %inserted_slice_289[9] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_294 = tensor.extract %inserted_slice_293[%c10] : tensor<32x!FHE.eint<2>>
    %extracted_295 = tensor.extract %arg3[%c10] : tensor<32x!FHE.eint<2>>
    %295 = "FHE.add_eint"(%extracted_294, %extracted_295) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %296 = "FHE.add_eint"(%295, %293) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %297 = "FHE.apply_lookup_table"(%296, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %298 = "FHE.apply_lookup_table"(%296, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_296 = tensor.from_elements %298 : tensor<1x!FHE.eint<2>>
    %inserted_slice_297 = tensor.insert_slice %from_elements_296 into %inserted_slice_293[10] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_298 = tensor.extract %inserted_slice_297[%c11] : tensor<32x!FHE.eint<2>>
    %extracted_299 = tensor.extract %arg3[%c11] : tensor<32x!FHE.eint<2>>
    %299 = "FHE.add_eint"(%extracted_298, %extracted_299) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %300 = "FHE.add_eint"(%299, %297) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %301 = "FHE.apply_lookup_table"(%300, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %302 = "FHE.apply_lookup_table"(%300, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_300 = tensor.from_elements %302 : tensor<1x!FHE.eint<2>>
    %inserted_slice_301 = tensor.insert_slice %from_elements_300 into %inserted_slice_297[11] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_302 = tensor.extract %inserted_slice_301[%c12] : tensor<32x!FHE.eint<2>>
    %extracted_303 = tensor.extract %arg3[%c12] : tensor<32x!FHE.eint<2>>
    %303 = "FHE.add_eint"(%extracted_302, %extracted_303) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %304 = "FHE.add_eint"(%303, %301) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %305 = "FHE.apply_lookup_table"(%304, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %306 = "FHE.apply_lookup_table"(%304, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_304 = tensor.from_elements %306 : tensor<1x!FHE.eint<2>>
    %inserted_slice_305 = tensor.insert_slice %from_elements_304 into %inserted_slice_301[12] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_306 = tensor.extract %inserted_slice_305[%c13] : tensor<32x!FHE.eint<2>>
    %extracted_307 = tensor.extract %arg3[%c13] : tensor<32x!FHE.eint<2>>
    %307 = "FHE.add_eint"(%extracted_306, %extracted_307) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %308 = "FHE.add_eint"(%307, %305) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %309 = "FHE.apply_lookup_table"(%308, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %310 = "FHE.apply_lookup_table"(%308, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_308 = tensor.from_elements %310 : tensor<1x!FHE.eint<2>>
    %inserted_slice_309 = tensor.insert_slice %from_elements_308 into %inserted_slice_305[13] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_310 = tensor.extract %inserted_slice_309[%c14] : tensor<32x!FHE.eint<2>>
    %extracted_311 = tensor.extract %arg3[%c14] : tensor<32x!FHE.eint<2>>
    %311 = "FHE.add_eint"(%extracted_310, %extracted_311) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %312 = "FHE.add_eint"(%311, %309) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %313 = "FHE.apply_lookup_table"(%312, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %314 = "FHE.apply_lookup_table"(%312, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_312 = tensor.from_elements %314 : tensor<1x!FHE.eint<2>>
    %inserted_slice_313 = tensor.insert_slice %from_elements_312 into %inserted_slice_309[14] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_314 = tensor.extract %inserted_slice_313[%c15] : tensor<32x!FHE.eint<2>>
    %extracted_315 = tensor.extract %arg3[%c15] : tensor<32x!FHE.eint<2>>
    %315 = "FHE.add_eint"(%extracted_314, %extracted_315) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %316 = "FHE.add_eint"(%315, %313) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %317 = "FHE.apply_lookup_table"(%316, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %318 = "FHE.apply_lookup_table"(%316, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_316 = tensor.from_elements %318 : tensor<1x!FHE.eint<2>>
    %inserted_slice_317 = tensor.insert_slice %from_elements_316 into %inserted_slice_313[15] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_318 = tensor.extract %inserted_slice_317[%c16] : tensor<32x!FHE.eint<2>>
    %extracted_319 = tensor.extract %arg3[%c16] : tensor<32x!FHE.eint<2>>
    %319 = "FHE.add_eint"(%extracted_318, %extracted_319) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %320 = "FHE.add_eint"(%319, %317) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %321 = "FHE.apply_lookup_table"(%320, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %322 = "FHE.apply_lookup_table"(%320, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_320 = tensor.from_elements %322 : tensor<1x!FHE.eint<2>>
    %inserted_slice_321 = tensor.insert_slice %from_elements_320 into %inserted_slice_317[16] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_322 = tensor.extract %inserted_slice_321[%c17] : tensor<32x!FHE.eint<2>>
    %extracted_323 = tensor.extract %arg3[%c17] : tensor<32x!FHE.eint<2>>
    %323 = "FHE.add_eint"(%extracted_322, %extracted_323) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %324 = "FHE.add_eint"(%323, %321) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %325 = "FHE.apply_lookup_table"(%324, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %326 = "FHE.apply_lookup_table"(%324, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_324 = tensor.from_elements %326 : tensor<1x!FHE.eint<2>>
    %inserted_slice_325 = tensor.insert_slice %from_elements_324 into %inserted_slice_321[17] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_326 = tensor.extract %inserted_slice_325[%c18] : tensor<32x!FHE.eint<2>>
    %extracted_327 = tensor.extract %arg3[%c18] : tensor<32x!FHE.eint<2>>
    %327 = "FHE.add_eint"(%extracted_326, %extracted_327) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %328 = "FHE.add_eint"(%327, %325) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %329 = "FHE.apply_lookup_table"(%328, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %330 = "FHE.apply_lookup_table"(%328, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_328 = tensor.from_elements %330 : tensor<1x!FHE.eint<2>>
    %inserted_slice_329 = tensor.insert_slice %from_elements_328 into %inserted_slice_325[18] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_330 = tensor.extract %inserted_slice_329[%c19] : tensor<32x!FHE.eint<2>>
    %extracted_331 = tensor.extract %arg3[%c19] : tensor<32x!FHE.eint<2>>
    %331 = "FHE.add_eint"(%extracted_330, %extracted_331) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %332 = "FHE.add_eint"(%331, %329) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %333 = "FHE.apply_lookup_table"(%332, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %334 = "FHE.apply_lookup_table"(%332, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_332 = tensor.from_elements %334 : tensor<1x!FHE.eint<2>>
    %inserted_slice_333 = tensor.insert_slice %from_elements_332 into %inserted_slice_329[19] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_334 = tensor.extract %inserted_slice_333[%c20] : tensor<32x!FHE.eint<2>>
    %extracted_335 = tensor.extract %arg3[%c20] : tensor<32x!FHE.eint<2>>
    %335 = "FHE.add_eint"(%extracted_334, %extracted_335) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %336 = "FHE.add_eint"(%335, %333) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %337 = "FHE.apply_lookup_table"(%336, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %338 = "FHE.apply_lookup_table"(%336, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_336 = tensor.from_elements %338 : tensor<1x!FHE.eint<2>>
    %inserted_slice_337 = tensor.insert_slice %from_elements_336 into %inserted_slice_333[20] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_338 = tensor.extract %inserted_slice_337[%c21] : tensor<32x!FHE.eint<2>>
    %extracted_339 = tensor.extract %arg3[%c21] : tensor<32x!FHE.eint<2>>
    %339 = "FHE.add_eint"(%extracted_338, %extracted_339) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %340 = "FHE.add_eint"(%339, %337) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %341 = "FHE.apply_lookup_table"(%340, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %342 = "FHE.apply_lookup_table"(%340, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_340 = tensor.from_elements %342 : tensor<1x!FHE.eint<2>>
    %inserted_slice_341 = tensor.insert_slice %from_elements_340 into %inserted_slice_337[21] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_342 = tensor.extract %inserted_slice_341[%c22] : tensor<32x!FHE.eint<2>>
    %extracted_343 = tensor.extract %arg3[%c22] : tensor<32x!FHE.eint<2>>
    %343 = "FHE.add_eint"(%extracted_342, %extracted_343) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %344 = "FHE.add_eint"(%343, %341) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %345 = "FHE.apply_lookup_table"(%344, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %346 = "FHE.apply_lookup_table"(%344, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_344 = tensor.from_elements %346 : tensor<1x!FHE.eint<2>>
    %inserted_slice_345 = tensor.insert_slice %from_elements_344 into %inserted_slice_341[22] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_346 = tensor.extract %inserted_slice_345[%c23] : tensor<32x!FHE.eint<2>>
    %extracted_347 = tensor.extract %arg3[%c23] : tensor<32x!FHE.eint<2>>
    %347 = "FHE.add_eint"(%extracted_346, %extracted_347) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %348 = "FHE.add_eint"(%347, %345) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %349 = "FHE.apply_lookup_table"(%348, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %350 = "FHE.apply_lookup_table"(%348, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_348 = tensor.from_elements %350 : tensor<1x!FHE.eint<2>>
    %inserted_slice_349 = tensor.insert_slice %from_elements_348 into %inserted_slice_345[23] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_350 = tensor.extract %inserted_slice_349[%c24] : tensor<32x!FHE.eint<2>>
    %extracted_351 = tensor.extract %arg3[%c24] : tensor<32x!FHE.eint<2>>
    %351 = "FHE.add_eint"(%extracted_350, %extracted_351) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %352 = "FHE.add_eint"(%351, %349) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %353 = "FHE.apply_lookup_table"(%352, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %354 = "FHE.apply_lookup_table"(%352, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_352 = tensor.from_elements %354 : tensor<1x!FHE.eint<2>>
    %inserted_slice_353 = tensor.insert_slice %from_elements_352 into %inserted_slice_349[24] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_354 = tensor.extract %inserted_slice_353[%c25] : tensor<32x!FHE.eint<2>>
    %extracted_355 = tensor.extract %arg3[%c25] : tensor<32x!FHE.eint<2>>
    %355 = "FHE.add_eint"(%extracted_354, %extracted_355) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %356 = "FHE.add_eint"(%355, %353) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %357 = "FHE.apply_lookup_table"(%356, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %358 = "FHE.apply_lookup_table"(%356, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_356 = tensor.from_elements %358 : tensor<1x!FHE.eint<2>>
    %inserted_slice_357 = tensor.insert_slice %from_elements_356 into %inserted_slice_353[25] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_358 = tensor.extract %inserted_slice_357[%c26] : tensor<32x!FHE.eint<2>>
    %extracted_359 = tensor.extract %arg3[%c26] : tensor<32x!FHE.eint<2>>
    %359 = "FHE.add_eint"(%extracted_358, %extracted_359) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %360 = "FHE.add_eint"(%359, %357) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %361 = "FHE.apply_lookup_table"(%360, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %362 = "FHE.apply_lookup_table"(%360, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_360 = tensor.from_elements %362 : tensor<1x!FHE.eint<2>>
    %inserted_slice_361 = tensor.insert_slice %from_elements_360 into %inserted_slice_357[26] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_362 = tensor.extract %inserted_slice_361[%c27] : tensor<32x!FHE.eint<2>>
    %extracted_363 = tensor.extract %arg3[%c27] : tensor<32x!FHE.eint<2>>
    %363 = "FHE.add_eint"(%extracted_362, %extracted_363) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %364 = "FHE.add_eint"(%363, %361) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %365 = "FHE.apply_lookup_table"(%364, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %366 = "FHE.apply_lookup_table"(%364, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_364 = tensor.from_elements %366 : tensor<1x!FHE.eint<2>>
    %inserted_slice_365 = tensor.insert_slice %from_elements_364 into %inserted_slice_361[27] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_366 = tensor.extract %inserted_slice_365[%c28] : tensor<32x!FHE.eint<2>>
    %extracted_367 = tensor.extract %arg3[%c28] : tensor<32x!FHE.eint<2>>
    %367 = "FHE.add_eint"(%extracted_366, %extracted_367) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %368 = "FHE.add_eint"(%367, %365) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %369 = "FHE.apply_lookup_table"(%368, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %370 = "FHE.apply_lookup_table"(%368, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_368 = tensor.from_elements %370 : tensor<1x!FHE.eint<2>>
    %inserted_slice_369 = tensor.insert_slice %from_elements_368 into %inserted_slice_365[28] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_370 = tensor.extract %inserted_slice_369[%c29] : tensor<32x!FHE.eint<2>>
    %extracted_371 = tensor.extract %arg3[%c29] : tensor<32x!FHE.eint<2>>
    %371 = "FHE.add_eint"(%extracted_370, %extracted_371) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %372 = "FHE.add_eint"(%371, %369) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %373 = "FHE.apply_lookup_table"(%372, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %374 = "FHE.apply_lookup_table"(%372, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_372 = tensor.from_elements %374 : tensor<1x!FHE.eint<2>>
    %inserted_slice_373 = tensor.insert_slice %from_elements_372 into %inserted_slice_369[29] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_374 = tensor.extract %inserted_slice_373[%c30] : tensor<32x!FHE.eint<2>>
    %extracted_375 = tensor.extract %arg3[%c30] : tensor<32x!FHE.eint<2>>
    %375 = "FHE.add_eint"(%extracted_374, %extracted_375) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %376 = "FHE.add_eint"(%375, %373) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %377 = "FHE.apply_lookup_table"(%376, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %378 = "FHE.apply_lookup_table"(%376, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_376 = tensor.from_elements %378 : tensor<1x!FHE.eint<2>>
    %inserted_slice_377 = tensor.insert_slice %from_elements_376 into %inserted_slice_373[30] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_378 = tensor.extract %inserted_slice_377[%c31] : tensor<32x!FHE.eint<2>>
    %extracted_379 = tensor.extract %arg3[%c31] : tensor<32x!FHE.eint<2>>
    %379 = "FHE.add_eint"(%extracted_378, %extracted_379) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %380 = "FHE.add_eint"(%379, %377) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %381 = "FHE.apply_lookup_table"(%380, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_380 = tensor.from_elements %381 : tensor<1x!FHE.eint<2>>
    %inserted_slice_381 = tensor.insert_slice %from_elements_380 into %inserted_slice_377[31] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_382 = tensor.extract %inserted_slice_381[%c0] : tensor<32x!FHE.eint<2>>
    %extracted_383 = tensor.extract %arg4[%c0] : tensor<32x!FHE.eint<2>>
    %382 = "FHE.add_eint"(%extracted_382, %extracted_383) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %383 = "FHE.add_eint_int"(%382, %c0_i2) : (!FHE.eint<2>, i2) -> !FHE.eint<2>
    %384 = "FHE.apply_lookup_table"(%383, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %385 = "FHE.apply_lookup_table"(%383, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_384 = tensor.from_elements %385 : tensor<1x!FHE.eint<2>>
    %inserted_slice_385 = tensor.insert_slice %from_elements_384 into %inserted_slice_381[0] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_386 = tensor.extract %inserted_slice_385[%c1] : tensor<32x!FHE.eint<2>>
    %extracted_387 = tensor.extract %arg4[%c1] : tensor<32x!FHE.eint<2>>
    %386 = "FHE.add_eint"(%extracted_386, %extracted_387) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %387 = "FHE.add_eint"(%386, %384) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %388 = "FHE.apply_lookup_table"(%387, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %389 = "FHE.apply_lookup_table"(%387, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_388 = tensor.from_elements %389 : tensor<1x!FHE.eint<2>>
    %inserted_slice_389 = tensor.insert_slice %from_elements_388 into %inserted_slice_385[1] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_390 = tensor.extract %inserted_slice_389[%c2] : tensor<32x!FHE.eint<2>>
    %extracted_391 = tensor.extract %arg4[%c2] : tensor<32x!FHE.eint<2>>
    %390 = "FHE.add_eint"(%extracted_390, %extracted_391) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %391 = "FHE.add_eint"(%390, %388) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %392 = "FHE.apply_lookup_table"(%391, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %393 = "FHE.apply_lookup_table"(%391, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_392 = tensor.from_elements %393 : tensor<1x!FHE.eint<2>>
    %inserted_slice_393 = tensor.insert_slice %from_elements_392 into %inserted_slice_389[2] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_394 = tensor.extract %inserted_slice_393[%c3] : tensor<32x!FHE.eint<2>>
    %extracted_395 = tensor.extract %arg4[%c3] : tensor<32x!FHE.eint<2>>
    %394 = "FHE.add_eint"(%extracted_394, %extracted_395) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %395 = "FHE.add_eint"(%394, %392) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %396 = "FHE.apply_lookup_table"(%395, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %397 = "FHE.apply_lookup_table"(%395, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_396 = tensor.from_elements %397 : tensor<1x!FHE.eint<2>>
    %inserted_slice_397 = tensor.insert_slice %from_elements_396 into %inserted_slice_393[3] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_398 = tensor.extract %inserted_slice_397[%c4] : tensor<32x!FHE.eint<2>>
    %extracted_399 = tensor.extract %arg4[%c4] : tensor<32x!FHE.eint<2>>
    %398 = "FHE.add_eint"(%extracted_398, %extracted_399) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %399 = "FHE.add_eint"(%398, %396) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %400 = "FHE.apply_lookup_table"(%399, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %401 = "FHE.apply_lookup_table"(%399, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_400 = tensor.from_elements %401 : tensor<1x!FHE.eint<2>>
    %inserted_slice_401 = tensor.insert_slice %from_elements_400 into %inserted_slice_397[4] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_402 = tensor.extract %inserted_slice_401[%c5] : tensor<32x!FHE.eint<2>>
    %extracted_403 = tensor.extract %arg4[%c5] : tensor<32x!FHE.eint<2>>
    %402 = "FHE.add_eint"(%extracted_402, %extracted_403) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %403 = "FHE.add_eint"(%402, %400) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %404 = "FHE.apply_lookup_table"(%403, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %405 = "FHE.apply_lookup_table"(%403, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_404 = tensor.from_elements %405 : tensor<1x!FHE.eint<2>>
    %inserted_slice_405 = tensor.insert_slice %from_elements_404 into %inserted_slice_401[5] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_406 = tensor.extract %inserted_slice_405[%c6] : tensor<32x!FHE.eint<2>>
    %extracted_407 = tensor.extract %arg4[%c6] : tensor<32x!FHE.eint<2>>
    %406 = "FHE.add_eint"(%extracted_406, %extracted_407) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %407 = "FHE.add_eint"(%406, %404) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %408 = "FHE.apply_lookup_table"(%407, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %409 = "FHE.apply_lookup_table"(%407, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_408 = tensor.from_elements %409 : tensor<1x!FHE.eint<2>>
    %inserted_slice_409 = tensor.insert_slice %from_elements_408 into %inserted_slice_405[6] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_410 = tensor.extract %inserted_slice_409[%c7] : tensor<32x!FHE.eint<2>>
    %extracted_411 = tensor.extract %arg4[%c7] : tensor<32x!FHE.eint<2>>
    %410 = "FHE.add_eint"(%extracted_410, %extracted_411) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %411 = "FHE.add_eint"(%410, %408) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %412 = "FHE.apply_lookup_table"(%411, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %413 = "FHE.apply_lookup_table"(%411, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_412 = tensor.from_elements %413 : tensor<1x!FHE.eint<2>>
    %inserted_slice_413 = tensor.insert_slice %from_elements_412 into %inserted_slice_409[7] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_414 = tensor.extract %inserted_slice_413[%c8] : tensor<32x!FHE.eint<2>>
    %extracted_415 = tensor.extract %arg4[%c8] : tensor<32x!FHE.eint<2>>
    %414 = "FHE.add_eint"(%extracted_414, %extracted_415) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %415 = "FHE.add_eint"(%414, %412) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %416 = "FHE.apply_lookup_table"(%415, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %417 = "FHE.apply_lookup_table"(%415, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_416 = tensor.from_elements %417 : tensor<1x!FHE.eint<2>>
    %inserted_slice_417 = tensor.insert_slice %from_elements_416 into %inserted_slice_413[8] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_418 = tensor.extract %inserted_slice_417[%c9] : tensor<32x!FHE.eint<2>>
    %extracted_419 = tensor.extract %arg4[%c9] : tensor<32x!FHE.eint<2>>
    %418 = "FHE.add_eint"(%extracted_418, %extracted_419) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %419 = "FHE.add_eint"(%418, %416) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %420 = "FHE.apply_lookup_table"(%419, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %421 = "FHE.apply_lookup_table"(%419, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_420 = tensor.from_elements %421 : tensor<1x!FHE.eint<2>>
    %inserted_slice_421 = tensor.insert_slice %from_elements_420 into %inserted_slice_417[9] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_422 = tensor.extract %inserted_slice_421[%c10] : tensor<32x!FHE.eint<2>>
    %extracted_423 = tensor.extract %arg4[%c10] : tensor<32x!FHE.eint<2>>
    %422 = "FHE.add_eint"(%extracted_422, %extracted_423) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %423 = "FHE.add_eint"(%422, %420) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %424 = "FHE.apply_lookup_table"(%423, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %425 = "FHE.apply_lookup_table"(%423, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_424 = tensor.from_elements %425 : tensor<1x!FHE.eint<2>>
    %inserted_slice_425 = tensor.insert_slice %from_elements_424 into %inserted_slice_421[10] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_426 = tensor.extract %inserted_slice_425[%c11] : tensor<32x!FHE.eint<2>>
    %extracted_427 = tensor.extract %arg4[%c11] : tensor<32x!FHE.eint<2>>
    %426 = "FHE.add_eint"(%extracted_426, %extracted_427) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %427 = "FHE.add_eint"(%426, %424) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %428 = "FHE.apply_lookup_table"(%427, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %429 = "FHE.apply_lookup_table"(%427, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_428 = tensor.from_elements %429 : tensor<1x!FHE.eint<2>>
    %inserted_slice_429 = tensor.insert_slice %from_elements_428 into %inserted_slice_425[11] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_430 = tensor.extract %inserted_slice_429[%c12] : tensor<32x!FHE.eint<2>>
    %extracted_431 = tensor.extract %arg4[%c12] : tensor<32x!FHE.eint<2>>
    %430 = "FHE.add_eint"(%extracted_430, %extracted_431) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %431 = "FHE.add_eint"(%430, %428) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %432 = "FHE.apply_lookup_table"(%431, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %433 = "FHE.apply_lookup_table"(%431, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_432 = tensor.from_elements %433 : tensor<1x!FHE.eint<2>>
    %inserted_slice_433 = tensor.insert_slice %from_elements_432 into %inserted_slice_429[12] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_434 = tensor.extract %inserted_slice_433[%c13] : tensor<32x!FHE.eint<2>>
    %extracted_435 = tensor.extract %arg4[%c13] : tensor<32x!FHE.eint<2>>
    %434 = "FHE.add_eint"(%extracted_434, %extracted_435) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %435 = "FHE.add_eint"(%434, %432) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %436 = "FHE.apply_lookup_table"(%435, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %437 = "FHE.apply_lookup_table"(%435, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_436 = tensor.from_elements %437 : tensor<1x!FHE.eint<2>>
    %inserted_slice_437 = tensor.insert_slice %from_elements_436 into %inserted_slice_433[13] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_438 = tensor.extract %inserted_slice_437[%c14] : tensor<32x!FHE.eint<2>>
    %extracted_439 = tensor.extract %arg4[%c14] : tensor<32x!FHE.eint<2>>
    %438 = "FHE.add_eint"(%extracted_438, %extracted_439) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %439 = "FHE.add_eint"(%438, %436) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %440 = "FHE.apply_lookup_table"(%439, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %441 = "FHE.apply_lookup_table"(%439, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_440 = tensor.from_elements %441 : tensor<1x!FHE.eint<2>>
    %inserted_slice_441 = tensor.insert_slice %from_elements_440 into %inserted_slice_437[14] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_442 = tensor.extract %inserted_slice_441[%c15] : tensor<32x!FHE.eint<2>>
    %extracted_443 = tensor.extract %arg4[%c15] : tensor<32x!FHE.eint<2>>
    %442 = "FHE.add_eint"(%extracted_442, %extracted_443) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %443 = "FHE.add_eint"(%442, %440) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %444 = "FHE.apply_lookup_table"(%443, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %445 = "FHE.apply_lookup_table"(%443, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_444 = tensor.from_elements %445 : tensor<1x!FHE.eint<2>>
    %inserted_slice_445 = tensor.insert_slice %from_elements_444 into %inserted_slice_441[15] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_446 = tensor.extract %inserted_slice_445[%c16] : tensor<32x!FHE.eint<2>>
    %extracted_447 = tensor.extract %arg4[%c16] : tensor<32x!FHE.eint<2>>
    %446 = "FHE.add_eint"(%extracted_446, %extracted_447) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %447 = "FHE.add_eint"(%446, %444) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %448 = "FHE.apply_lookup_table"(%447, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %449 = "FHE.apply_lookup_table"(%447, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_448 = tensor.from_elements %449 : tensor<1x!FHE.eint<2>>
    %inserted_slice_449 = tensor.insert_slice %from_elements_448 into %inserted_slice_445[16] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_450 = tensor.extract %inserted_slice_449[%c17] : tensor<32x!FHE.eint<2>>
    %extracted_451 = tensor.extract %arg4[%c17] : tensor<32x!FHE.eint<2>>
    %450 = "FHE.add_eint"(%extracted_450, %extracted_451) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %451 = "FHE.add_eint"(%450, %448) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %452 = "FHE.apply_lookup_table"(%451, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %453 = "FHE.apply_lookup_table"(%451, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_452 = tensor.from_elements %453 : tensor<1x!FHE.eint<2>>
    %inserted_slice_453 = tensor.insert_slice %from_elements_452 into %inserted_slice_449[17] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_454 = tensor.extract %inserted_slice_453[%c18] : tensor<32x!FHE.eint<2>>
    %extracted_455 = tensor.extract %arg4[%c18] : tensor<32x!FHE.eint<2>>
    %454 = "FHE.add_eint"(%extracted_454, %extracted_455) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %455 = "FHE.add_eint"(%454, %452) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %456 = "FHE.apply_lookup_table"(%455, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %457 = "FHE.apply_lookup_table"(%455, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_456 = tensor.from_elements %457 : tensor<1x!FHE.eint<2>>
    %inserted_slice_457 = tensor.insert_slice %from_elements_456 into %inserted_slice_453[18] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_458 = tensor.extract %inserted_slice_457[%c19] : tensor<32x!FHE.eint<2>>
    %extracted_459 = tensor.extract %arg4[%c19] : tensor<32x!FHE.eint<2>>
    %458 = "FHE.add_eint"(%extracted_458, %extracted_459) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %459 = "FHE.add_eint"(%458, %456) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %460 = "FHE.apply_lookup_table"(%459, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %461 = "FHE.apply_lookup_table"(%459, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_460 = tensor.from_elements %461 : tensor<1x!FHE.eint<2>>
    %inserted_slice_461 = tensor.insert_slice %from_elements_460 into %inserted_slice_457[19] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_462 = tensor.extract %inserted_slice_461[%c20] : tensor<32x!FHE.eint<2>>
    %extracted_463 = tensor.extract %arg4[%c20] : tensor<32x!FHE.eint<2>>
    %462 = "FHE.add_eint"(%extracted_462, %extracted_463) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %463 = "FHE.add_eint"(%462, %460) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %464 = "FHE.apply_lookup_table"(%463, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %465 = "FHE.apply_lookup_table"(%463, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_464 = tensor.from_elements %465 : tensor<1x!FHE.eint<2>>
    %inserted_slice_465 = tensor.insert_slice %from_elements_464 into %inserted_slice_461[20] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_466 = tensor.extract %inserted_slice_465[%c21] : tensor<32x!FHE.eint<2>>
    %extracted_467 = tensor.extract %arg4[%c21] : tensor<32x!FHE.eint<2>>
    %466 = "FHE.add_eint"(%extracted_466, %extracted_467) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %467 = "FHE.add_eint"(%466, %464) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %468 = "FHE.apply_lookup_table"(%467, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %469 = "FHE.apply_lookup_table"(%467, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_468 = tensor.from_elements %469 : tensor<1x!FHE.eint<2>>
    %inserted_slice_469 = tensor.insert_slice %from_elements_468 into %inserted_slice_465[21] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_470 = tensor.extract %inserted_slice_469[%c22] : tensor<32x!FHE.eint<2>>
    %extracted_471 = tensor.extract %arg4[%c22] : tensor<32x!FHE.eint<2>>
    %470 = "FHE.add_eint"(%extracted_470, %extracted_471) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %471 = "FHE.add_eint"(%470, %468) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %472 = "FHE.apply_lookup_table"(%471, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %473 = "FHE.apply_lookup_table"(%471, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_472 = tensor.from_elements %473 : tensor<1x!FHE.eint<2>>
    %inserted_slice_473 = tensor.insert_slice %from_elements_472 into %inserted_slice_469[22] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_474 = tensor.extract %inserted_slice_473[%c23] : tensor<32x!FHE.eint<2>>
    %extracted_475 = tensor.extract %arg4[%c23] : tensor<32x!FHE.eint<2>>
    %474 = "FHE.add_eint"(%extracted_474, %extracted_475) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %475 = "FHE.add_eint"(%474, %472) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %476 = "FHE.apply_lookup_table"(%475, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %477 = "FHE.apply_lookup_table"(%475, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_476 = tensor.from_elements %477 : tensor<1x!FHE.eint<2>>
    %inserted_slice_477 = tensor.insert_slice %from_elements_476 into %inserted_slice_473[23] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_478 = tensor.extract %inserted_slice_477[%c24] : tensor<32x!FHE.eint<2>>
    %extracted_479 = tensor.extract %arg4[%c24] : tensor<32x!FHE.eint<2>>
    %478 = "FHE.add_eint"(%extracted_478, %extracted_479) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %479 = "FHE.add_eint"(%478, %476) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %480 = "FHE.apply_lookup_table"(%479, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %481 = "FHE.apply_lookup_table"(%479, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_480 = tensor.from_elements %481 : tensor<1x!FHE.eint<2>>
    %inserted_slice_481 = tensor.insert_slice %from_elements_480 into %inserted_slice_477[24] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_482 = tensor.extract %inserted_slice_481[%c25] : tensor<32x!FHE.eint<2>>
    %extracted_483 = tensor.extract %arg4[%c25] : tensor<32x!FHE.eint<2>>
    %482 = "FHE.add_eint"(%extracted_482, %extracted_483) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %483 = "FHE.add_eint"(%482, %480) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %484 = "FHE.apply_lookup_table"(%483, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %485 = "FHE.apply_lookup_table"(%483, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_484 = tensor.from_elements %485 : tensor<1x!FHE.eint<2>>
    %inserted_slice_485 = tensor.insert_slice %from_elements_484 into %inserted_slice_481[25] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_486 = tensor.extract %inserted_slice_485[%c26] : tensor<32x!FHE.eint<2>>
    %extracted_487 = tensor.extract %arg4[%c26] : tensor<32x!FHE.eint<2>>
    %486 = "FHE.add_eint"(%extracted_486, %extracted_487) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %487 = "FHE.add_eint"(%486, %484) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %488 = "FHE.apply_lookup_table"(%487, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %489 = "FHE.apply_lookup_table"(%487, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_488 = tensor.from_elements %489 : tensor<1x!FHE.eint<2>>
    %inserted_slice_489 = tensor.insert_slice %from_elements_488 into %inserted_slice_485[26] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_490 = tensor.extract %inserted_slice_489[%c27] : tensor<32x!FHE.eint<2>>
    %extracted_491 = tensor.extract %arg4[%c27] : tensor<32x!FHE.eint<2>>
    %490 = "FHE.add_eint"(%extracted_490, %extracted_491) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %491 = "FHE.add_eint"(%490, %488) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %492 = "FHE.apply_lookup_table"(%491, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %493 = "FHE.apply_lookup_table"(%491, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_492 = tensor.from_elements %493 : tensor<1x!FHE.eint<2>>
    %inserted_slice_493 = tensor.insert_slice %from_elements_492 into %inserted_slice_489[27] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_494 = tensor.extract %inserted_slice_493[%c28] : tensor<32x!FHE.eint<2>>
    %extracted_495 = tensor.extract %arg4[%c28] : tensor<32x!FHE.eint<2>>
    %494 = "FHE.add_eint"(%extracted_494, %extracted_495) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %495 = "FHE.add_eint"(%494, %492) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %496 = "FHE.apply_lookup_table"(%495, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %497 = "FHE.apply_lookup_table"(%495, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_496 = tensor.from_elements %497 : tensor<1x!FHE.eint<2>>
    %inserted_slice_497 = tensor.insert_slice %from_elements_496 into %inserted_slice_493[28] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_498 = tensor.extract %inserted_slice_497[%c29] : tensor<32x!FHE.eint<2>>
    %extracted_499 = tensor.extract %arg4[%c29] : tensor<32x!FHE.eint<2>>
    %498 = "FHE.add_eint"(%extracted_498, %extracted_499) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %499 = "FHE.add_eint"(%498, %496) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %500 = "FHE.apply_lookup_table"(%499, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %501 = "FHE.apply_lookup_table"(%499, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_500 = tensor.from_elements %501 : tensor<1x!FHE.eint<2>>
    %inserted_slice_501 = tensor.insert_slice %from_elements_500 into %inserted_slice_497[29] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_502 = tensor.extract %inserted_slice_501[%c30] : tensor<32x!FHE.eint<2>>
    %extracted_503 = tensor.extract %arg4[%c30] : tensor<32x!FHE.eint<2>>
    %502 = "FHE.add_eint"(%extracted_502, %extracted_503) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %503 = "FHE.add_eint"(%502, %500) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %504 = "FHE.apply_lookup_table"(%503, %cst) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %505 = "FHE.apply_lookup_table"(%503, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_504 = tensor.from_elements %505 : tensor<1x!FHE.eint<2>>
    %inserted_slice_505 = tensor.insert_slice %from_elements_504 into %inserted_slice_501[30] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_506 = tensor.extract %inserted_slice_505[%c31] : tensor<32x!FHE.eint<2>>
    %extracted_507 = tensor.extract %arg4[%c31] : tensor<32x!FHE.eint<2>>
    %506 = "FHE.add_eint"(%extracted_506, %extracted_507) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %507 = "FHE.add_eint"(%506, %504) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
    %508 = "FHE.apply_lookup_table"(%507, %cst_1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
    %from_elements_508 = tensor.from_elements %508 : tensor<1x!FHE.eint<2>>
    %inserted_slice_509 = tensor.insert_slice %from_elements_508 into %inserted_slice_505[31] [1] [1] : tensor<1x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    return %inserted_slice_509 : tensor<32x!FHE.eint<2>>
  }
```
<<<<<<< HEAD

=======
</details>

- `iftern` function
<details>
>>>>>>> 78decb21 (docs(frontend): adding a SHA1 tutorial with modules)
```
  func.func @iftern(%arg0: tensor<32x!FHE.eint<2>>, %arg1: tensor<32x!FHE.eint<2>>, %arg2: tensor<32x!FHE.eint<2>>) -> tensor<32x!FHE.eint<2>> {
    %c2_i3 = arith.constant 2 : i3
    %from_elements = tensor.from_elements %c2_i3 : tensor<1xi3>
    %0 = "FHELinalg.mul_eint_int"(%arg1, %from_elements) : (tensor<32x!FHE.eint<2>>, tensor<1xi3>) -> tensor<32x!FHE.eint<2>>
    %1 = "FHELinalg.add_eint"(%0, %arg2) : (tensor<32x!FHE.eint<2>>, tensor<32x!FHE.eint<2>>) -> tensor<32x!FHE.eint<2>>
    %cst = arith.constant dense<[0, 1, 1, 0]> : tensor<4xi64>
    %2 = "FHELinalg.apply_lookup_table"(%1, %cst) : (tensor<32x!FHE.eint<2>>, tensor<4xi64>) -> tensor<32x!FHE.eint<2>>
    %3 = "FHELinalg.mul_eint_int"(%arg0, %from_elements) : (tensor<32x!FHE.eint<2>>, tensor<1xi3>) -> tensor<32x!FHE.eint<2>>
    %4 = "FHELinalg.add_eint"(%3, %2) : (tensor<32x!FHE.eint<2>>, tensor<32x!FHE.eint<2>>) -> tensor<32x!FHE.eint<2>>
    %cst_0 = arith.constant dense<[0, 0, 0, 1]> : tensor<4xi64>
    %5 = "FHELinalg.apply_lookup_table"(%4, %cst_0) : (tensor<32x!FHE.eint<2>>, tensor<4xi64>) -> tensor<32x!FHE.eint<2>>
    %6 = "FHELinalg.mul_eint_int"(%arg2, %from_elements) : (tensor<32x!FHE.eint<2>>, tensor<1xi3>) -> tensor<32x!FHE.eint<2>>
    %7 = "FHELinalg.add_eint"(%6, %5) : (tensor<32x!FHE.eint<2>>, tensor<32x!FHE.eint<2>>) -> tensor<32x!FHE.eint<2>>
    %8 = "FHELinalg.apply_lookup_table"(%7, %cst) : (tensor<32x!FHE.eint<2>>, tensor<4xi64>) -> tensor<32x!FHE.eint<2>>
    return %8 : tensor<32x!FHE.eint<2>>
  }
```
<<<<<<< HEAD

=======
</details>

- `maj` function
<details>
>>>>>>> 78decb21 (docs(frontend): adding a SHA1 tutorial with modules)
```
  func.func @maj(%arg0: tensor<32x!FHE.eint<2>>, %arg1: tensor<32x!FHE.eint<2>>, %arg2: tensor<32x!FHE.eint<2>>) -> tensor<32x!FHE.eint<2>> {
    %c2_i3 = arith.constant 2 : i3
    %from_elements = tensor.from_elements %c2_i3 : tensor<1xi3>
    %0 = "FHELinalg.mul_eint_int"(%arg0, %from_elements) : (tensor<32x!FHE.eint<2>>, tensor<1xi3>) -> tensor<32x!FHE.eint<2>>
    %1 = "FHELinalg.add_eint"(%0, %arg1) : (tensor<32x!FHE.eint<2>>, tensor<32x!FHE.eint<2>>) -> tensor<32x!FHE.eint<2>>
    %cst = arith.constant dense<[0, 0, 0, 1]> : tensor<4xi64>
    %2 = "FHELinalg.apply_lookup_table"(%1, %cst) : (tensor<32x!FHE.eint<2>>, tensor<4xi64>) -> tensor<32x!FHE.eint<2>>
    %cst_0 = arith.constant dense<[0, 1, 1, 1]> : tensor<4xi64>
    %3 = "FHELinalg.apply_lookup_table"(%1, %cst_0) : (tensor<32x!FHE.eint<2>>, tensor<4xi64>) -> tensor<32x!FHE.eint<2>>
    %4 = "FHELinalg.mul_eint_int"(%arg2, %from_elements) : (tensor<32x!FHE.eint<2>>, tensor<1xi3>) -> tensor<32x!FHE.eint<2>>
    %5 = "FHELinalg.add_eint"(%4, %3) : (tensor<32x!FHE.eint<2>>, tensor<32x!FHE.eint<2>>) -> tensor<32x!FHE.eint<2>>
    %6 = "FHELinalg.apply_lookup_table"(%5, %cst) : (tensor<32x!FHE.eint<2>>, tensor<4xi64>) -> tensor<32x!FHE.eint<2>>
    %7 = "FHELinalg.mul_eint_int"(%2, %from_elements) : (tensor<32x!FHE.eint<2>>, tensor<1xi3>) -> tensor<32x!FHE.eint<2>>
    %8 = "FHELinalg.add_eint"(%7, %6) : (tensor<32x!FHE.eint<2>>, tensor<32x!FHE.eint<2>>) -> tensor<32x!FHE.eint<2>>
    %9 = "FHELinalg.apply_lookup_table"(%8, %cst_0) : (tensor<32x!FHE.eint<2>>, tensor<4xi64>) -> tensor<32x!FHE.eint<2>>
    return %9 : tensor<32x!FHE.eint<2>>
  }
```
<<<<<<< HEAD

=======
</details>

- `rotate30` function
<details>
>>>>>>> 78decb21 (docs(frontend): adding a SHA1 tutorial with modules)
```
  func.func @rotate30(%arg0: tensor<32x!FHE.eint<2>>) -> tensor<32x!FHE.eint<2>> {
    %0 = "FHE.zero_tensor"() : () -> tensor<32x!FHE.eint<2>>
    %extracted_slice = tensor.extract_slice %arg0[0] [2] [1] : tensor<32x!FHE.eint<2>> to tensor<2x!FHE.eint<2>>
    %inserted_slice = tensor.insert_slice %extracted_slice into %0[30] [2] [1] : tensor<2x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_slice_0 = tensor.extract_slice %arg0[2] [30] [1] : tensor<32x!FHE.eint<2>> to tensor<30x!FHE.eint<2>>
    %inserted_slice_1 = tensor.insert_slice %extracted_slice_0 into %inserted_slice[0] [30] [1] : tensor<30x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    return %inserted_slice_1 : tensor<32x!FHE.eint<2>>
  }
```
<<<<<<< HEAD

=======
</details>

- `rotate5` function
<details>
>>>>>>> 78decb21 (docs(frontend): adding a SHA1 tutorial with modules)
```
  func.func @rotate5(%arg0: tensor<32x!FHE.eint<2>>) -> tensor<32x!FHE.eint<2>> {
    %0 = "FHE.zero_tensor"() : () -> tensor<32x!FHE.eint<2>>
    %extracted_slice = tensor.extract_slice %arg0[0] [27] [1] : tensor<32x!FHE.eint<2>> to tensor<27x!FHE.eint<2>>
    %inserted_slice = tensor.insert_slice %extracted_slice into %0[5] [27] [1] : tensor<27x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    %extracted_slice_0 = tensor.extract_slice %arg0[27] [5] [1] : tensor<32x!FHE.eint<2>> to tensor<5x!FHE.eint<2>>
    %inserted_slice_1 = tensor.insert_slice %extracted_slice_0 into %inserted_slice[0] [5] [1] : tensor<5x!FHE.eint<2>> into tensor<32x!FHE.eint<2>>
    return %inserted_slice_1 : tensor<32x!FHE.eint<2>>
  }
```
<<<<<<< HEAD

=======
</details>

- `xor3` function
<details>
>>>>>>> 78decb21 (docs(frontend): adding a SHA1 tutorial with modules)
```
  func.func @xor3(%arg0: tensor<32x!FHE.eint<2>>, %arg1: tensor<32x!FHE.eint<2>>, %arg2: tensor<32x!FHE.eint<2>>) -> tensor<32x!FHE.eint<2>> {
    %c2_i3 = arith.constant 2 : i3
    %from_elements = tensor.from_elements %c2_i3 : tensor<1xi3>
    %0 = "FHELinalg.mul_eint_int"(%arg0, %from_elements) : (tensor<32x!FHE.eint<2>>, tensor<1xi3>) -> tensor<32x!FHE.eint<2>>
    %1 = "FHELinalg.add_eint"(%0, %arg1) : (tensor<32x!FHE.eint<2>>, tensor<32x!FHE.eint<2>>) -> tensor<32x!FHE.eint<2>>
    %cst = arith.constant dense<[0, 1, 1, 0]> : tensor<4xi64>
    %2 = "FHELinalg.apply_lookup_table"(%1, %cst) : (tensor<32x!FHE.eint<2>>, tensor<4xi64>) -> tensor<32x!FHE.eint<2>>
    %3 = "FHELinalg.mul_eint_int"(%2, %from_elements) : (tensor<32x!FHE.eint<2>>, tensor<1xi3>) -> tensor<32x!FHE.eint<2>>
    %4 = "FHELinalg.add_eint"(%3, %arg2) : (tensor<32x!FHE.eint<2>>, tensor<32x!FHE.eint<2>>) -> tensor<32x!FHE.eint<2>>
    %5 = "FHELinalg.apply_lookup_table"(%4, %cst) : (tensor<32x!FHE.eint<2>>, tensor<4xi64>) -> tensor<32x!FHE.eint<2>>
    return %5 : tensor<32x!FHE.eint<2>>
  }
}
```
<<<<<<< HEAD
=======
</details>
>>>>>>> 78decb21 (docs(frontend): adding a SHA1 tutorial with modules)

## Testing or Using

Once again, this tutorial is more to show the use of Modules than using it in production. For a full
client-server API, one may want to do more in FHE, including the message expansion, and maybe
optimize more the functions.

One can check that the implementation works in FHE by running `python sha1.py --autotest`: it will
pick a certain number of random inputs, hash them in FHE and compare the result with the `hashlib`
standard implementation.

<<<<<<< HEAD
One can also hash a given value with:
``
=======
One can also hash a given value with
`echo -n "The quick brown fox jumps over the lazy dog" | python sha1.py`, and it will print
something like:

```
sha1-digest: 2fd4e1c67a2d28fced849ee1bb76e7391b93eb12
computed in: 320.265383 seconds
```

## Benchmarks

We have executed our implementation on an HPC7a machine with Concrete 2.7.0rc1.

`python sha1.py --autotest` typically returns:

```
Checking SHA1(fdASguUMBwhPcKuDpPqoRlQXLrLQbnxEvPJSQSIUDTBoaqrJlBualgoWEINmDZDYSuGuSOpGBWwWzjAfktWYZZUliv) for an input length 90
sha1-digest: 5bb539fd423875ccc8a33148dae724f5b2cf9391
computed in: 295.306287 seconds
Checking SHA1(BYwXTbqE) for an input length 8
sha1-digest: 90a8dcad6ddff7ca8fd487b80a37fcd250c56bed
computed in: 145.341164 seconds
Checking SHA1(rnPZh) for an input length 5
sha1-digest: 47610d2c26ee8b45ab0f4c8f8e4d405b2cd37f1f
computed in: 145.318081 seconds
Checking SHA1(orRaJMGbUJtxITQvqiOCPjKJWYuHomuiexCQQgZyTeAAFJcgCftDCRAkcLKjRECelIMPQphGEUlSNthE) for an input length 80
sha1-digest: bd74b4e64349d308f3b95b54cf61ee416bdd6b18
computed in: 288.240576 seconds
Checking SHA1(ROokDcdczajNPjlCPoWotaRJHBtOVyiyxMIIeCtxaDCjk) for an input length 45
sha1-digest: 1ff546c3a64f27339781c095cbc097f392c2cccd
computed in: 143.621941 seconds
Checking SHA1(KbCXFt) for an input length 6
sha1-digest: 7e5789f0c83fa5102004fbeeef3ac22244d1cdac
computed in: 143.509567 seconds
Checking SHA1(mpKnkHtrgokxgQSzcIjFtxKnhmMfZbIbkJavnkSxW) for an input length 41
sha1-digest: 1308d9f7cba634ab2617edb5116b8bdf434f16f5
computed in: 143.341450 seconds
Checking SHA1(oauoWKJGyjjTcXqRIxFGuVuMwiwjKYfttQ) for an input length 34
sha1-digest: 60367153b7049ca92eb979ad7b809c5a3f47a64e
computed in: 143.693254 seconds
Checking SHA1(ZMGiaIOmBJPncOsUCxj) for an input length 19
sha1-digest: fafba9f2fe6b5a0fddad4ad765909c8fc32117c6
computed in: 143.720215 seconds
Checking SHA1(HwCXIHnFoGUgIBqaQrrpDnhEvPBX) for an input length 28
sha1-digest: 5224cace20f8d20fa3ea8d9974b5ff3a0be7fd48
computed in: 143.523006 seconds
Checking SHA1(AfyzsimngrqeWoqZKOBRwVuvttfgJTpegMbiHjUNdWzTg) for an input length 45
sha1-digest: 8ca27aca1c362ca63e50d58aa7065b4322f028a0
computed in: 143.481069 seconds
Checking SHA1(hNEUPakrqQpGGZvtHvht) for an input length 20
sha1-digest: 36ae34ed85e62ac0f922e36fc98b23e725695be1
computed in: 143.478666 seconds
Checking SHA1(CjgfYYlNKqZdHeXFfqTwhycbGBeSpzpxKPwWItriiNKZCcEJRZlM) for an input length 52
sha1-digest: 3c012f41c5fe4581f80e2901fc4bbbb70ff7a9ba
computed in: 143.490262 seconds
Checking SHA1(EXIGkYzWpcqpfRKCSbBJJqqmUBkFwWfPGooJvsVAshWjMr) for an input length 46
sha1-digest: 2518c4d13ec7608f59632ac993b726e572c3aaae
computed in: 143.840785 seconds
Checking SHA1(sgzaAqZnhXmFJOJMyfGxweYFMmLeUHmMCWETfqzstzpFYKaGpnasiLHPTcJtukHztEQpXzquREcbtoJDaoqjfM) for an input length 86
sha1-digest: 46f4b0653ed7ea0ce89cc18f6720e5e334d63a45
computed in: 288.155301 seconds
Checking SHA1(oRaisdHJovDxCnwyComEGejqMceBTOVhJucVnwgC) for an input length 40
sha1-digest: 909f9c6275aa9f41d8ecaf52203bb0e24cf978d7
computed in: 143.466817 seconds
Checking SHA1(mtTWxtHerQgLdBGftWdiCwBKqtu) for an input length 27
sha1-digest: 624a7dcec460061a2a6499dae978fe4afd674110
computed in: 145.389956 seconds
Checking SHA1(beYzkJLvZMmoXbQwqoVThpyaQ) for an input length 25
sha1-digest: 25a9df47bd055384a9ee614c1dc7213c04f2087c
computed in: 147.234881 seconds
Checking SHA1(CpQWXXRNlXIoSZNxmXUwWHqmUAdlOrDyZPzzOhznlpGntrUgvktlZ) for an input length 53
sha1-digest: f2bde6574d8f6aa360929f6a5f919700b16e093b
computed in: 147.154393 seconds
Checking SHA1(busWigrVdsXnkjTh) for an input length 16
sha1-digest: fe47568d433278a38a4729f7891d03eaacdb0e40
computed in: 147.465694 seconds
Checking SHA1()
sha1-digest: da39a3ee5e6b4b0d3255bfef95601890afd80709
computed in: 147.256297 seconds
Checking SHA1(The quick brown fox jumps over the lazy dog)
sha1-digest: 2fd4e1c67a2d28fced849ee1bb76e7391b93eb12
computed in: 147.697102 seconds
```

means that:
- one block of compression takes about 147 seconds
- two blocks of compression take about 290 seconds


>>>>>>> 78decb21 (docs(frontend): adding a SHA1 tutorial with modules)















