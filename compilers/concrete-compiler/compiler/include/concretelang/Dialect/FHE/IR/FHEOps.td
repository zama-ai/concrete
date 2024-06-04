//===- FHEOps.td - High level FHE dialect ops ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONCRETELANG_DIALECT_FHE_IR_FHE_OPS
#define CONCRETELANG_DIALECT_FHE_IR_FHE_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

include "concretelang/Dialect/FHE/IR/FHEDialect.td"
include "concretelang/Dialect/FHE/IR/FHETypes.td"
include "concretelang/Dialect/FHE/IR/FHEAttrs.td"
include "concretelang/Dialect/FHE/Interfaces/FHEInterfaces.td"

class FHE_Op<string mnemonic, list<Trait> traits = []> :
    Op<FHE_Dialect, mnemonic, traits>;

def FHE_ZeroEintOp : FHE_Op<"zero", [Pure, ConstantNoise]> {
    let summary = "Returns a trivial encrypted integer of 0";

    let description = [{
        Example:
        ```mlir
        "FHE.zero"() : () -> !FHE.eint<2>
        "FHE.zero"() : () -> !FHE.esint<2>
        ```
    }];

    let arguments = (ins);
    let results = (outs FHE_AnyEncryptedInteger:$out);
}

def FHE_ZeroTensorOp : FHE_Op<"zero_tensor", [Pure, ConstantNoise]> {
    let summary = "Creates a new tensor with all elements initialized to an encrypted zero.";

    let description = [{
        Creates a new tensor with the shape specified in the result type and initializes its elements with an encrypted zero.

        Example:
        ```mlir
        %tensor = "FHE.zero_tensor"() : () -> tensor<5x!FHE.eint<4>>
        %tensor = "FHE.zero_tensor"() : () -> tensor<5x!FHE.esint<4>>
        ```
    }];

    let arguments = (ins);

    let results = (outs Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>:$tensor);
}

def FHE_AddEintIntOp : FHE_Op<"add_eint_int", [Pure, BinaryEintInt, DeclareOpInterfaceMethods<Binary>]> {
    let summary = "Adds an encrypted integer and a clear integer";

    let description = [{
        The clear integer must have at most one more bit than the encrypted integer
        and the result must have the same width and the same signedness as the encrypted integer.

        Example:
        ```mlir
        // ok
        "FHE.add_eint_int"(%a, %i) : (!FHE.eint<2>, i3) -> !FHE.eint<2>
        "FHE.add_eint_int"(%a, %i) : (!FHE.esint<2>, i3) -> !FHE.esint<2>

        // error
        "FHE.add_eint_int"(%a, %i) : (!FHE.eint<2>, i4) -> !FHE.eint<2>
        "FHE.add_eint_int"(%a, %i) : (!FHE.eint<2>, i3) -> !FHE.eint<3>
        "FHE.add_eint_int"(%a, %i) : (!FHE.eint<2>, i3) -> !FHE.esint<2>
        ```
    }];

    let arguments = (ins FHE_AnyEncryptedInteger:$a, AnyInteger:$b);
    let results = (outs FHE_AnyEncryptedInteger);

    let builders = [
        OpBuilder<(ins "Value":$a, "Value":$b), [{
         build($_builder, $_state, a.getType(), a, b);
        }]>
    ];

    let hasVerifier = 1;
    let hasFolder = 1;
}

def FHE_AddEintOp : FHE_Op<"add_eint", [Pure, BinaryEint, DeclareOpInterfaceMethods<BinaryEint>]> {
    let summary = "Adds two encrypted integers";

    let description = [{
        The encrypted integers and the result must have the same width and the same signedness.

        Example:
        ```mlir
        // ok
        "FHE.add_eint"(%a, %b): (!FHE.eint<2>, !FHE.eint<2>) -> (!FHE.eint<2>)
        "FHE.add_eint"(%a, %b): (!FHE.esint<2>, !FHE.esint<2>) -> (!FHE.esint<2>)

        // error
        "FHE.add_eint"(%a, %b): (!FHE.eint<2>, !FHE.eint<3>) -> (!FHE.eint<2>)
        "FHE.add_eint"(%a, %b): (!FHE.eint<2>, !FHE.eint<2>) -> (!FHE.eint<3>)
        "FHE.add_eint"(%a, %b): (!FHE.eint<2>, !FHE.eint<2>) -> (!FHE.esint<2>)
        "FHE.add_eint"(%a, %b): (!FHE.esint<2>, !FHE.eint<2>) -> (!FHE.eint<2>)
        ```
    }];

    let arguments = (ins FHE_AnyEncryptedInteger:$a, FHE_AnyEncryptedInteger:$b);
    let results = (outs FHE_AnyEncryptedInteger);

    let builders = [
        OpBuilder<(ins "Value":$a, "Value":$b), [{
         build($_builder, $_state, a.getType(), a, b);
        }]>
    ];

    let hasVerifier = 1;
}

def FHE_SubIntEintOp : FHE_Op<"sub_int_eint", [Pure, BinaryIntEint]> {
    let summary = "Subtract an encrypted integer from a clear integer";

    let description = [{
        The clear integer must have one more bit than the encrypted integer
        and the result must have the same width and the same signedness as the encrypted integer.

        Example:
        ```mlir
        // ok
        "FHE.sub_int_eint"(%i, %a) : (i3, !FHE.eint<2>) -> !FHE.eint<2>
        "FHE.sub_int_eint"(%i, %a) : (i3, !FHE.esint<2>) -> !FHE.esint<2>

        // error
        "FHE.sub_int_eint"(%i, %a) : (i4, !FHE.eint<2>) -> !FHE.eint<2>
        "FHE.sub_int_eint"(%i, %a) : (i3, !FHE.eint<2>) -> !FHE.eint<3>
        "FHE.sub_int_eint"(%i, %a) : (i3, !FHE.eint<2>) -> !FHE.esint<2>
        ```
    }];

    let arguments = (ins AnyInteger:$a, FHE_AnyEncryptedInteger:$b);
    let results = (outs FHE_AnyEncryptedInteger);

    let builders = [
        OpBuilder<(ins "Value":$a, "Value":$b), [{
         build($_builder, $_state, b.getType(), a, b);
        }]>
    ];

    let hasVerifier = 1;
}

def FHE_SubEintIntOp : FHE_Op<"sub_eint_int", [Pure, BinaryEintInt, DeclareOpInterfaceMethods<Binary>]> {
    let summary = "Subtract a clear integer from an encrypted integer";

    let description = [{
        The clear integer must have one more bit than the encrypted integer
        and the result must have the same width and the same signedness as the encrypted integer.

        Example:
        ```mlir
        // ok
        "FHE.sub_eint_int"(%i, %a) : (!FHE.eint<2>, i3) -> !FHE.eint<2>
        "FHE.sub_eint_int"(%i, %a) : (!FHE.esint<2>, i3) -> !FHE.esint<2>

        // error
        "FHE.sub_eint_int"(%i, %a) : (!FHE.eint<2>, i4) -> !FHE.eint<2>
        "FHE.sub_eint_int"(%i, %a) : (!FHE.eint<2>, i3) -> !FHE.eint<3>
        "FHE.sub_eint_int"(%i, %a) : (!FHE.eint<2>, i3) -> !FHE.esint<2>
        ```
    }];

    let arguments = (ins FHE_AnyEncryptedInteger:$a, AnyInteger:$b);
    let results = (outs FHE_AnyEncryptedInteger);

    let builders = [
        OpBuilder<(ins "Value":$a, "Value":$b), [{
            build($_builder, $_state, a.getType(), a, b);
        }]>
    ];

    let hasVerifier = 1;
    let hasFolder = 1;
}

def FHE_SubEintOp : FHE_Op<"sub_eint", [Pure, BinaryEint, DeclareOpInterfaceMethods<BinaryEint>]> {
    let summary = "Subtract an encrypted integer from an encrypted integer";

    let description = [{
        The encrypted integers and the result must have the same width and the same signedness.

        Example:
        ```mlir
        // ok
        "FHE.sub_eint"(%a, %b): (!FHE.eint<2>, !FHE.eint<2>) -> (!FHE.eint<2>)
        "FHE.sub_eint"(%a, %b): (!FHE.esint<2>, !FHE.esint<2>) -> (!FHE.esint<2>)

        // error
        "FHE.sub_eint"(%a, %b): (!FHE.eint<2>, !FHE.eint<3>) -> (!FHE.eint<2>)
        "FHE.sub_eint"(%a, %b): (!FHE.eint<2>, !FHE.eint<2>) -> (!FHE.eint<3>)
        "FHE.sub_eint"(%a, %b): (!FHE.eint<2>, !FHE.esint<2>) -> (!FHE.esint<2>)
        "FHE.sub_eint"(%a, %b): (!FHE.eint<2>, !FHE.eint<2>) -> (!FHE.esint<2>)
        ```
    }];

    let arguments = (ins FHE_AnyEncryptedInteger:$a, FHE_AnyEncryptedInteger:$b);
    let results = (outs FHE_AnyEncryptedInteger);

    let builders = [
        OpBuilder<(ins "Value":$a, "Value":$b), [{
            build($_builder, $_state, a.getType(), a, b);
        }]>
    ];

    let hasVerifier = 1;
}

def FHE_NegEintOp : FHE_Op<"neg_eint", [Pure, UnaryEint, DeclareOpInterfaceMethods<UnaryEint>]> {

    let summary = "Negates an encrypted integer";

    let description = [{
        The result must have the same width and the same signedness as the encrypted integer.

        Example:
        ```mlir
        // ok
        "FHE.neg_eint"(%a): (!FHE.eint<2>) -> (!FHE.eint<2>)
        "FHE.neg_eint"(%a): (!FHE.esint<2>) -> (!FHE.esint<2>)

        // error
        "FHE.neg_eint"(%a): (!FHE.eint<2>) -> (!FHE.eint<3>)
        "FHE.neg_eint"(%a): (!FHE.eint<2>) -> (!FHE.esint<2>)
        ```
    }];

    let arguments = (ins FHE_AnyEncryptedInteger:$a);
    let results = (outs FHE_AnyEncryptedInteger);

    let builders = [
        OpBuilder<(ins "Value":$a), [{
         build($_builder, $_state, a.getType(), a);
        }]>
    ];
    let hasVerifier = 1;
}

def FHE_MulEintIntOp : FHE_Op<"mul_eint_int", [Pure, BinaryEintInt, DeclareOpInterfaceMethods<Binary, ["sqMANP"]>]> {
    let summary = "Multiply an encrypted integer with a clear integer";

    let description = [{
        The clear integer must have one more bit than the encrypted integer
        and the result must have the same width and the same signedness as the encrypted integer.

        Example:
        ```mlir
        // ok
        "FHE.mul_eint_int"(%a, %i) : (!FHE.eint<2>, i3) -> !FHE.eint<2>
        "FHE.mul_eint_int"(%a, %i) : (!FHE.esint<2>, i3) -> !FHE.esint<2>

        // error
        "FHE.mul_eint_int"(%a, %i) : (!FHE.eint<2>, i4) -> !FHE.eint<2>
        "FHE.mul_eint_int"(%a, %i) : (!FHE.eint<2>, i3) -> !FHE.eint<3>
        "FHE.mul_eint_int"(%a, %i) : (!FHE.eint<2>, i3) -> !FHE.esint<2>
        ```
    }];

    let arguments = (ins FHE_AnyEncryptedInteger:$a, AnyInteger:$b);
    let results = (outs FHE_AnyEncryptedInteger);

    let builders = [
        OpBuilder<(ins "Value":$a, "Value":$b), [{
         build($_builder, $_state, a.getType(), a, b);
        }]>
    ];

    let hasVerifier = 1;
    let hasFolder = 1;
    let hasCanonicalizer = 1;
}

def FHE_MulEintOp : FHE_Op<"mul_eint", [Pure, BinaryEint, DeclareOpInterfaceMethods<BinaryEint>]> {
    let summary = "Multiplies two encrypted integers";

    let description = [{
        The encrypted integers and the result must have the same width and
        signedness. Also, due to the current implementation, one supplementary
        bit of width must be provided, in addition to the number of bits needed
        to encode the largest output value.

        Example:
        ```mlir
        // ok
        "FHE.mul_eint"(%a, %b): (!FHE.eint<2>, !FHE.eint<2>) -> (!FHE.eint<2>)
        "FHE.mul_eint"(%a, %b): (!FHE.eint<3>, !FHE.eint<3>) -> (!FHE.eint<3>)
        "FHE.mul_eint"(%a, %b): (!FHE.esint<3>, !FHE.esint<3>) -> (!FHE.esint<3>)

        // error
        "FHE.mul_eint"(%a, %b): (!FHE.eint<2>, !FHE.eint<3>) -> (!FHE.eint<2>)
        "FHE.mul_eint"(%a, %b): (!FHE.eint<2>, !FHE.eint<2>) -> (!FHE.eint<3>)
        "FHE.mul_eint"(%a, %b): (!FHE.eint<2>, !FHE.eint<2>) -> (!FHE.esint<2>)
        "FHE.mul_eint"(%a, %b): (!FHE.esint<2>, !FHE.eint<2>) -> (!FHE.eint<2>)
        ```
    }];

    let arguments = (ins FHE_AnyEncryptedInteger:$rhs, FHE_AnyEncryptedInteger:$lhs);
    let results = (outs FHE_AnyEncryptedInteger);

    let builders = [
        OpBuilder<(ins "Value":$rhs, "Value":$lhs), [{
          build($_builder, $_state, rhs.getType(), rhs, lhs);
        }]>
    ];

    let hasVerifier = 1;
}

def FHE_MaxEintOp : FHE_Op<"max_eint", [Pure, BinaryEint, DeclareOpInterfaceMethods<BinaryEint>]> {
    let summary = "Retrieve the maximum of two encrypted integers.";

    let description = [{
        Retrieve the maximum of two encrypted integers using the formula, 'max(x, y) == max(x - y, 0) + y'.
        The input and output types should be the same.

        If `x - y`` inside the max overflows or underflows, the behavior is undefined.
        To support the full range, you should increase the bit-width by 1 manually.

        Example:
        ```mlir
        // ok
        "FHE.max_eint"(%x, %y) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
        "FHE.max_eint"(%x, %y) : (!FHE.esint<3>, !FHE.esint<3>) -> !FHE.esint<3>

        // error
        "FHE.max_eint"(%x, %y) : (!FHE.eint<2>, !FHE.eint<3>) -> !FHE.eint<2>
        "FHE.max_eint"(%x, %y) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.esint<2>
        "FHE.max_eint"(%x, %y) : (!FHE.esint<2>, !FHE.eint<2>) -> !FHE.eint<2>
        ```
    }];

    let arguments = (ins FHE_AnyEncryptedInteger:$x, FHE_AnyEncryptedInteger:$y);
    let results = (outs FHE_AnyEncryptedInteger);

    let builders = [
        OpBuilder<(ins "Value":$x, "Value":$y), [{
         build($_builder, $_state, x.getType(), x, y);
        }]>
    ];

    let hasVerifier = 1;
}

def FHE_ToSignedOp : FHE_Op<"to_signed", [Pure, UnaryEint, DeclareOpInterfaceMethods<UnaryEint>]> {
    let summary = "Cast an unsigned integer to a signed one";

    let description = [{
        The result must have the same width as the input.

        The behavior is undefined on overflow/underflow.

        Examples:
        ```mlir
        // ok
        "FHE.to_signed"(%x) : (!FHE.eint<2>) -> !FHE.esint<2>

        // error
        "FHE.to_signed"(%x) : (!FHE.eint<2>) -> !FHE.esint<3>
        ```
    }];

    let arguments = (ins FHE_EncryptedUnsignedIntegerType:$input);
    let results = (outs FHE_EncryptedSignedIntegerType);

    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

def FHE_ToUnsignedOp : FHE_Op<"to_unsigned", [Pure, UnaryEint, DeclareOpInterfaceMethods<UnaryEint>]> {
    let summary = "Cast a signed integer to an unsigned one";

    let description = [{
        The result must have the same width as the input.

        The behavior is undefined on overflow/underflow.

        Examples:
        ```mlir
        // ok
        "FHE.to_unsigned"(%x) : (!FHE.esint<2>) -> !FHE.eint<2>

        // error
        "FHE.to_unsigned"(%x) : (!FHE.esint<2>) -> !FHE.eint<3>
        ```
    }];

    let arguments = (ins FHE_EncryptedSignedIntegerType:$input);
    let results = (outs FHE_EncryptedUnsignedIntegerType);

    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

def FHE_ApplyLookupTableEintOp : FHE_Op<"apply_lookup_table", [Pure, ConstantNoise]> {

    let summary = "Applies a clear lookup table to an encrypted integer";

    let description = [{
        The width of the result can be different than the width of the operand.
        The lookup table must be a tensor of size `2^p` where `p` is the width of the encrypted integer.

        Example:
        ```mlir
        // ok
        "FHE.apply_lookup_table"(%a, %lut): (!FHE.eint<2>, tensor<4xi64>) -> (!FHE.eint<2>)
        "FHE.apply_lookup_table"(%a, %lut): (!FHE.eint<2>, tensor<4xi64>) -> (!FHE.eint<3>)
        "FHE.apply_lookup_table"(%a, %lut): (!FHE.eint<3>, tensor<4xi64>) -> (!FHE.eint<2>)

        // error
        "FHE.apply_lookup_table"(%a, %lut): (!FHE.eint<2>, tensor<8xi64>) -> (!FHE.eint<2>)
        ```
    }];

    let arguments = (ins FHE_AnyEncryptedInteger:$a,
        TensorOf<[AnyInteger]>:$lut);
    let results = (outs FHE_AnyEncryptedInteger);

    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

def FHE_RoundEintOp: FHE_Op<"round", [Pure, UnaryEint, DeclareOpInterfaceMethods<UnaryEint, ["sqMANP"]>]> {

    let summary = "Rounds a ciphertext to a smaller precision.";

    let description = [{
      Assuming a ciphertext whose message is implemented over `p` bits, this
          operation rounds it to fit to `q` bits with `p>q`.

      Example:
      ```mlir
       // ok
       "FHE.round"(%a): (!FHE.eint<6>) -> (!FHE.eint<5>)
       "FHE.round"(%a): (!FHE.eint<5>) -> (!FHE.eint<3>)
       "FHE.round"(%a): (!FHE.eint<3>) -> (!FHE.eint<2>)
       "FHE.round"(%a): (!FHE.esint<3>) -> (!FHE.esint<2>)

    // error
       "FHE.round"(%a): (!FHE.eint<6>) -> (!FHE.eint<6>)
       "FHE.round"(%a): (!FHE.eint<4>) -> (!FHE.eint<5>)
       "FHE.round"(%a): (!FHE.eint<4>) -> (!FHE.esint<5>)

    ```
    }];

    let arguments = (ins FHE_AnyEncryptedInteger:$input);
    let results = (outs FHE_AnyEncryptedInteger);
    let hasVerifier = 1;
    let hasFolder = 1;
}

def FHE_LsbEintOp: FHE_Op<"lsb", [Pure, UnaryEint, ConstantNoise]> {

    let summary = "Extract the lowest significant bit at a given precision.";

    let description = [{
      This operation extracts the lsb of a ciphertext in a specific precision.

      Extracting the lsb with the smallest precision:
      ```mlir
       // Checking if even or odd
       %even = "FHE.lsb"(%a): (!FHE.eint<4>) -> (!FHE.eint<1>)

      Usually when you extract the lsb bit, you also need to extract the next one.
      In that case you first need to clear the first lsb of the input to be able to reduce its precision and extract the next one.
      To be able to clear the lsb just extracted, you can get it in the original precision.

      Example:
      ```mlir
       // Extracting the first lsb with original precision
       %lsb_0 = "FHE.lsb"(%input): (!FHE.eint<4>) -> (!FHE.eint<4>)
       // Clearing the first lsb from original input
       %input_lsb0_cleared = "FHE.sub_eint"(%input, %lsb_0) : (!FHE.eint<4>, !FHE.eint<4>) -> (!FHE.eint<4>)
       // Reducing the precision of the input
       %input_3b = "FHE.reinterpret_precision(%input) : (!FHE.eint<4>) -> !FHE.eint<3>
       // Now, we can do it again with the second lsb
       %lsb_1 = "FHE.lsb"(%input_3b): (!FHE.eint<3>) -> (!FHE.eint<3>)
       ...
       // later if you need %b_lsb at same position as in the input
       %lsb_1_at_input_position = "FHE.reinterpret_precision(%b_lsb)" : (!FHE.eint<3>) -> !FHE.eint<4>
       // that way you can recombine the extracted bits
       %input_mod_4 = "FHE.add_eint"(%lsb_0, %lsb_1_at_input_position) : (!FHE.eint<4>, !FHE.eint<4>) -> (!FHE.eint<4>)
    ```
    }];

    let arguments = (ins FHE_AnyEncryptedInteger:$input);
    let results = (outs FHE_AnyEncryptedInteger);
}

def FHE_ReinterpretPrecisionEintOp: FHE_Op<"reinterpret_precision", [Pure, UnaryEint]> {

    let summary = "Reinterpret the ciphertext with a different precision.";

    let description = [{
      Changing the precision of a ciphertext.
      It changes both the precision, the value, and in certain cases the correctness of the ciphertext.

      Changing to
        - a bigger precision is always safe.
          This is equivalent to a shift left for the value.
        - a smaller precision is only safe if you clear the lowest bits that are discarded.
          If not, you can assume small errors on the next TLU.
          This is equivalent to a shift right for the value.

      Example:
      ```mlir
       // assuming %a is stored as 4bits but can be stored with only 2bits
       // we can reduce its storage precision
       %shifted_a = "FHE.mul_eint_int"(%a, %c_4): (!FHE.eint<4>) -> (!FHE.eint<4>)
       %a_small_precision = "FHE.reinterpret_precision"(%shifted_a, %lsb) : (!FHE.eint<4>) -> (!FHE.eint<2>)
    ```
    }];

    let arguments = (ins FHE_AnyEncryptedInteger:$input);
    let results = (outs FHE_AnyEncryptedInteger);
}

def FHE_ChangePartitionEintOp: FHE_Op<"change_partition", [Pure, UnaryEint]> {

    let summary = "Change partition if necessary.";

    let description = [{
      Changing the partition of a ciphertext.
      If necessary, it keyswitch the ciphertext to a different key having a different set of parameters than the original one.

      Example:
      ```mlir
        %new_eint = "FHE.change_partition"(%eint): (!FHE.eint<16>) -> (!FHE.eint<16>)
    ```
    }];

    let arguments = (ins FHE_AnyEncryptedInteger:$input, OptionalAttr<FHE_PartitionAttr>:$src, OptionalAttr<FHE_PartitionAttr>:$dest);
    let results = (outs FHE_AnyEncryptedInteger);
    let hasVerifier = 1;
}

// FHE Boolean Operations

def FHE_GenGateOp : FHE_Op<"gen_gate", [Pure]> {

    let summary = "Applies a truth table based on two boolean inputs";

    let description = [{
        Truth table must be a tensor of four boolean values.

        Example:
        ```mlir
        // ok
        "FHE.gen_gate"(%a, %b, %ttable): (!FHE.ebool, !FHE.ebool, tensor<4xi64>) -> (!FHE.ebool)

        // error
        "FHE.gen_gate"(%a, %b, %ttable): (!FHE.ebool, !FHE.ebool, tensor<7xi64>) -> (!FHE.ebool)
        ```
    }];

    // The reason the truth table is of AnyInteger and not I1 is that in lowering passes, the truth_table is meant to be passed
    // to an LUT operation which requires the table to be of type I64. Whenever lowering passes are no more restrictive, this
    // can be set to I1 to reflect the boolean logic.
    let arguments = (ins FHE_EncryptedBooleanType:$left, FHE_EncryptedBooleanType:$right, TensorOf<[AnyInteger]>:$truth_table);
    let results = (outs FHE_EncryptedBooleanType);
    let hasVerifier = 1;
}

def FHE_MuxOp : FHE_Op<"mux", [Pure]> {

    let summary = "Multiplexer for two encrypted boolean inputs, based on an encrypted condition";

    let description = [{
        Example:
        ```mlir
        "FHE.mux"(%cond, %c1, %c2): (!FHE.ebool, !FHE.ebool, !FHE.ebool) -> (!FHE.ebool)
        ```
    }];

    let arguments = (ins FHE_EncryptedBooleanType:$cond, FHE_EncryptedBooleanType:$c1, FHE_EncryptedBooleanType:$c2);
    let results = (outs FHE_EncryptedBooleanType);
}

def FHE_BoolAndOp : FHE_Op<"and", [Pure]> {

    let summary = "Applies an AND gate to two encrypted boolean values";

    let description = [{
        Example:
        ```mlir
        "FHE.and"(%a, %b): (!FHE.ebool, !FHE.ebool) -> (!FHE.ebool)
        ```
    }];

    let arguments = (ins FHE_EncryptedBooleanType:$left, FHE_EncryptedBooleanType:$right);
    let results = (outs FHE_EncryptedBooleanType);
}

def FHE_BoolOrOp : FHE_Op<"or", [Pure]> {

    let summary = "Applies an OR gate to two encrypted boolean values";

    let description = [{
        Example:
        ```mlir
        "FHE.or"(%a, %b): (!FHE.ebool, !FHE.ebool) -> (!FHE.ebool)
        ```
    }];

    let arguments = (ins FHE_EncryptedBooleanType:$left, FHE_EncryptedBooleanType:$right);
    let results = (outs FHE_EncryptedBooleanType);
}

def FHE_BoolNandOp : FHE_Op<"nand", [Pure]> {

    let summary = "Applies a NAND gate to two encrypted boolean values";

    let description = [{
        Example:
        ```mlir
        "FHE.nand"(%a, %b): (!FHE.ebool, !FHE.ebool) -> (!FHE.ebool)
        ```
    }];

    let arguments = (ins FHE_EncryptedBooleanType:$left, FHE_EncryptedBooleanType:$right);
    let results = (outs FHE_EncryptedBooleanType);
}

def FHE_BoolXorOp : FHE_Op<"xor", [Pure]> {

    let summary = "Applies an XOR gate to two encrypted boolean values";

    let description = [{
        Example:
        ```mlir
        "FHE.xor"(%a, %b): (!FHE.ebool, !FHE.ebool) -> (!FHE.ebool)
        ```
    }];

    let arguments = (ins FHE_EncryptedBooleanType:$left, FHE_EncryptedBooleanType:$right);
    let results = (outs FHE_EncryptedBooleanType);
}

def FHE_BoolNotOp : FHE_Op<"not", [Pure, UnaryEint, DeclareOpInterfaceMethods<UnaryEint>]> {

    let summary = "Applies a NOT gate to an encrypted boolean value";

    let description = [{
        Example:
        ```mlir
        "FHE.not"(%a): (!FHE.ebool) -> (!FHE.ebool)
        ```
    }];

    let arguments = (ins FHE_EncryptedBooleanType:$value);
    let results = (outs FHE_EncryptedBooleanType);
}

def FHE_ToBoolOp : FHE_Op<"to_bool", [Pure, UnaryEint]> {
    let summary = "Cast an unsigned integer to a boolean";

    let description = [{
        The input must be of width one or two. Two being the current representation
        of an encrypted boolean, leaving one bit for the carry.

        Examples:
        ```mlir
        // ok
        "FHE.to_bool"(%x) : (!FHE.eint<1>) -> !FHE.ebool
        "FHE.to_bool"(%x) : (!FHE.eint<2>) -> !FHE.ebool

        // error
        "FHE.to_bool"(%x) : (!FHE.eint<3>) -> !FHE.ebool
        ```
    }];

    let arguments = (ins FHE_EncryptedUnsignedIntegerType:$input);
    let results = (outs FHE_EncryptedBooleanType);

    let hasVerifier = 1;
}

def FHE_FromBoolOp : FHE_Op<"from_bool", [Pure, UnaryEint]> {
    let summary = "Cast a boolean to an unsigned integer";

    let description = [{
        Examples:
        ```mlir
        "FHE.from_bool"(%x) : (!FHE.ebool) -> !FHE.eint<1>
        "FHE.from_bool"(%x) : (!FHE.ebool) -> !FHE.eint<2>
        "FHE.from_bool"(%x) : (!FHE.ebool) -> !FHE.eint<4>
        ```
    }];

    let arguments = (ins FHE_EncryptedBooleanType:$input);
    let results = (outs FHE_EncryptedUnsignedIntegerType);
}



#endif
