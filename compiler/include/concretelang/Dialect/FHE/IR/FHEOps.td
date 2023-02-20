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

class FHE_Op<string mnemonic, list<Trait> traits = []> :
    Op<FHE_Dialect, mnemonic, traits>;

def FHE_ZeroEintOp : FHE_Op<"zero", [NoSideEffect]> {
    let summary = "Returns a trivial encrypted integer of 0";

    let description = [{
        Returns a trivial encrypted integer of 0

        Example:
        ```mlir
        "FHE.zero"() : () -> !FHE.eint<2>
        "FHE.zero"() : () -> !FHE.esint<2>
        ```
    }];

    let arguments = (ins);
    let results = (outs FHE_AnyEncryptedInteger:$out);
}

def FHE_ZeroTensorOp : FHE_Op<"zero_tensor", [NoSideEffect]> {
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

def FHE_AddEintIntOp : FHE_Op<"add_eint_int", [NoSideEffect]> {
    let summary = "Adds an encrypted integer and a clear integer";

    let description = [{
        Adds an encrypted integer and a clear integer.
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

def FHE_AddEintOp : FHE_Op<"add_eint", [NoSideEffect]> {
    let summary = "Adds two encrypted integers";

    let description = [{
        Adds two encrypted integers
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

def FHE_SubIntEintOp : FHE_Op<"sub_int_eint", [NoSideEffect]> {
    let summary = "Subtract an encrypted integer from a clear integer";

    let description = [{
        Subtract an encrypted integer from a clear integer.
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

def FHE_SubEintIntOp : FHE_Op<"sub_eint_int", [NoSideEffect]> {
    let summary = "Subtract a clear integer from an encrypted integer";

    let description = [{
        Subtract a clear integer from an encrypted integer.
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

def FHE_SubEintOp : FHE_Op<"sub_eint", [NoSideEffect]> {
    let summary = "Subtract an encrypted integer from an encrypted integer";

    let description = [{
        Subtract an encrypted integer from an encrypted integer.
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

def FHE_NegEintOp : FHE_Op<"neg_eint", [NoSideEffect]> {

    let summary = "Negates an encrypted integer";

    let description = [{
        Negates an encrypted integer.
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

def FHE_MulEintIntOp : FHE_Op<"mul_eint_int", [NoSideEffect]> {
    let summary = "Multiply an encrypted integer with a clear integer";

    let description = [{
        Multiply an encrypted integer with a clear integer.
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
}

def FHE_MulEintOp : FHE_Op<"mul_eint", [NoSideEffect]> {
    let summary = "Multiplies two encrypted integers";

    let description = [{
        Multiplies two encrypted integers.

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

    let arguments = (ins FHE_AnyEncryptedInteger:$a, FHE_AnyEncryptedInteger:$b);
    let results = (outs FHE_AnyEncryptedInteger);

    let builders = [
        OpBuilder<(ins "Value":$a, "Value":$b), [{
          build($_builder, $_state, a.getType(), a, b);
        }]>
    ];

    let hasVerifier = 1;
}

def FHE_ToSignedOp : FHE_Op<"to_signed", [NoSideEffect]> {
    let summary = "Cast an unsigned integer to a signed one";

    let description = [{
        Cast an unsigned integer to a signed one.
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

    let arguments = (ins FHE_EncryptedIntegerType:$input);
    let results = (outs FHE_EncryptedSignedIntegerType);

    let hasVerifier = 1;
}

def FHE_ToUnsignedOp : FHE_Op<"to_unsigned", [NoSideEffect]> {
    let summary = "Cast a signed integer to an unsigned one";

    let description = [{
        Cast a signed integer to an unsigned one.
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
    let results = (outs FHE_EncryptedIntegerType);

    let hasVerifier = 1;
}

def FHE_ApplyLookupTableEintOp : FHE_Op<"apply_lookup_table", [NoSideEffect]> {

    let summary = "Applies a clear lookup table to an encrypted integer";

    let description = [{
        Applies a clear lookup table to an encrypted integer, the width of the result can be different than the width of the operand.
        The lookup table must be a tensor of size equals to `2^p` where `p` is the width of the encrypted integer.

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
}

def FHE_RoundEintOp: FHE_Op<"round", [NoSideEffect]> {

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
}

// FHE Boolean Operations

def FHE_GenGateOp : FHE_Op<"gen_gate", [NoSideEffect]> {

    let summary = "Applies a truth table based on two boolean inputs";

    let description = [{
        Applies a truth table based on two boolean inputs.

        truth table must be a tensor of 4 boolean values.

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

def FHE_MuxOp : FHE_Op<"mux", [NoSideEffect]> {

    let summary = "Multiplexer for two encrypted boolean inputs, based on an encrypted condition";

    let description = [{
        Mutex between two encrypted boolean inputs, based on an encrypted condition.

        Example:
        ```mlir
        "FHE.mux"(%cond, %c1, %c2): (!FHE.ebool, !FHE.ebool, !FHE.ebool) -> (!FHE.ebool)
        ```
    }];

    let arguments = (ins FHE_EncryptedBooleanType:$cond, FHE_EncryptedBooleanType:$c1, FHE_EncryptedBooleanType:$c2);
    let results = (outs FHE_EncryptedBooleanType);
}

def FHE_BoolAndOp : FHE_Op<"and", [NoSideEffect]> {

    let summary = "Applies an AND gate to two encrypted boolean values";

    let description = [{
        Applies an AND gate to two encrypted boolean values.

        Example:
        ```mlir
        "FHE.and"(%a, %b): (!FHE.ebool, !FHE.ebool) -> (!FHE.ebool)
        ```
    }];

    let arguments = (ins FHE_EncryptedBooleanType:$left, FHE_EncryptedBooleanType:$right);
    let results = (outs FHE_EncryptedBooleanType);
}

def FHE_BoolOrOp : FHE_Op<"or", [NoSideEffect]> {

    let summary = "Applies an OR gate to two encrypted boolean values";

    let description = [{
        Applies an OR gate to two encrypted boolean values.

        Example:
        ```mlir
        "FHE.or"(%a, %b): (!FHE.ebool, !FHE.ebool) -> (!FHE.ebool)
        ```
    }];

    let arguments = (ins FHE_EncryptedBooleanType:$left, FHE_EncryptedBooleanType:$right);
    let results = (outs FHE_EncryptedBooleanType);
}

def FHE_BoolNandOp : FHE_Op<"nand", [NoSideEffect]> {

    let summary = "Applies a NAND gate to two encrypted boolean values";

    let description = [{
        Applies a NAND gate to two encrypted boolean values.

        Example:
        ```mlir
        "FHE.nand"(%a, %b): (!FHE.ebool, !FHE.ebool) -> (!FHE.ebool)
        ```
    }];

    let arguments = (ins FHE_EncryptedBooleanType:$left, FHE_EncryptedBooleanType:$right);
    let results = (outs FHE_EncryptedBooleanType);
}

def FHE_BoolXorOp : FHE_Op<"xor", [NoSideEffect]> {

    let summary = "Applies a XOR gate to two encrypted boolean values";

    let description = [{
        Applies a XOR gate to two encrypted boolean values.

        Example:
        ```mlir
        "FHE.xor"(%a, %b): (!FHE.ebool, !FHE.ebool) -> (!FHE.ebool)
        ```
    }];

    let arguments = (ins FHE_EncryptedBooleanType:$left, FHE_EncryptedBooleanType:$right);
    let results = (outs FHE_EncryptedBooleanType);
}

def FHE_BoolNotOp : FHE_Op<"not", [NoSideEffect]> {

    let summary = "Applies a NOT gate to an encrypted boolean value";

    let description = [{
        Applies a NOT gate to an encrypted boolean value.

        Example:
        ```mlir
        "FHE.not"(%a): (!FHE.ebool) -> (!FHE.ebool)
        ```
    }];

    let arguments = (ins FHE_EncryptedBooleanType:$value);
    let results = (outs FHE_EncryptedBooleanType);
}

def FHE_ToBoolOp : FHE_Op<"to_bool", [NoSideEffect]> {
    let summary = "Cast an unsigned integer to a boolean";

    let description = [{
        Cast an unsigned integer to a boolean.

        The input must necessarily be of width 1 or 2. 2 being the current representation
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

    let arguments = (ins FHE_EncryptedIntegerType:$input);
    let results = (outs FHE_EncryptedBooleanType);

    let hasVerifier = 1;
}

def FHE_FromBoolOp : FHE_Op<"from_bool", [NoSideEffect]> {
    let summary = "Cast a boolean to an unsigned integer";

    let description = [{
        Cast a boolean to an unsigned integer.

        Examples:
        ```mlir
        "FHE.from_bool"(%x) : (!FHE.ebool) -> !FHE.eint<1>
        "FHE.from_bool"(%x) : (!FHE.ebool) -> !FHE.eint<2>
        "FHE.from_bool"(%x) : (!FHE.ebool) -> !FHE.eint<4>
        ```
    }];

    let arguments = (ins FHE_EncryptedBooleanType:$input);
    let results = (outs FHE_EncryptedIntegerType);
}



#endif
