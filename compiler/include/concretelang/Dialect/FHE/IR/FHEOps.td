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

class FHE_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<FHE_Dialect, mnemonic, traits>;

def ZeroEintOp : FHE_Op<"zero", [NoSideEffect]> {
    let summary = "Returns a trivial encrypted integer of 0";

    let description = [{
        Returns a trivial encrypted integer of 0

        Example:
        ```mlir
        "FHE.zero"() : () -> !FHE.eint<2>
        ```
    }];

    let arguments = (ins);
    let results = (outs EncryptedIntegerType:$out);
}


def ZeroTensorOp : FHE_Op<"zero_tensor", []> {
    let summary = "Creates a new tensor with all elements initialized to an encrypted zero.";

    let description = [{
        Creates a new tensor with the shape specified in the result type and initializes its elements with an encrypted zero.

        Example:
        ```mlir
        %tensor = "FHE.zero_tensor"() : () -> tensor<5x!FHE.eint<4>>
        ```
    }];

    let arguments = (ins);

    let results = (outs Type<And<[TensorOf<[EncryptedIntegerType]>.predicate, HasStaticShapePred]>>:$tensor);
}

def AddEintIntOp : FHE_Op<"add_eint_int"> {

    let summary = "Adds an encrypted integer and a clear integer";

    let description = [{
        Adds an encrypted integer and a clear integer.
        The clear integer must have at most one more bit than the encrypted integer
        and the result must have the same width than the encrypted integer.

        Example:
        ```mlir
        // ok
        "FHE.add_eint_int"(%a, %i) : (!FHE.eint<2>, i3) -> !FHE.eint<2>

        // error
        "FHE.add_eint_int"(%a, %i) : (!FHE.eint<2>, i4) -> !FHE.eint<2>
        "FHE.add_eint_int"(%a, %i) : (!FHE.eint<2>, i3) -> !FHE.eint<3>
        ```
    }];

    let arguments = (ins EncryptedIntegerType:$a, AnyInteger:$b);
    let results = (outs EncryptedIntegerType);

    let builders = [
        OpBuilder<(ins "Value":$a, "Value":$b), [{
         build($_builder, $_state, a.getType(), a, b);
        }]>
    ];

    let verifier = [{
        return ::mlir::concretelang::FHE::verifyAddEintIntOp(*this);
    }];
}

def AddEintOp : FHE_Op<"add_eint"> {

    let summary = "Adds two encrypted integers";

        let description = [{
        Adds two encrypted integers
        The encrypted integers and the result must have the same width.

        Example:
        ```mlir
        // ok
        "FHE.add_eint"(%a, %b): (!FHE.eint<2>, !FHE.eint<2>) -> (!FHE.eint<2>)

        // error
        "FHE.add_eint"(%a, %b): (!FHE.eint<2>, !FHE.eint<3>) -> (!FHE.eint<2>)
        "FHE.add_eint"(%a, %b): (!FHE.eint<2>, !FHE.eint<2>) -> (!FHE.eint<3>)
        ```
    }];

    let arguments = (ins EncryptedIntegerType:$a, EncryptedIntegerType:$b);
    let results = (outs EncryptedIntegerType);

    let builders = [
        OpBuilder<(ins "Value":$a, "Value":$b), [{
         build($_builder, $_state, a.getType(), a, b);
        }]>
    ];

    let verifier = [{
        return ::mlir::concretelang::FHE::verifyAddEintOp(*this);
    }];
}

def SubIntEintOp : FHE_Op<"sub_int_eint"> {

    let summary = "Substract a clear integer and an encrypted integer";

    let description = [{
        Substract a clear integer and an encrypted integer.
        The clear integer must have at most one more bit than the encrypted integer
        and the result must have the same width than the encrypted integer.

        Example:
        ```mlir
        // ok
        "FHE.sub_int_eint"(%i, %a) : (i3, !FHE.eint<2>) -> !FHE.eint<2>

        // error
        "FHE.sub_int_eint"(%i, %a) : (i4, !FHE.eint<2>) -> !FHE.eint<2>
        "FHE.sub_int_eint"(%i, %a) : (i3, !FHE.eint<2>) -> !FHE.eint<3>
        ```
    }];

    let arguments = (ins AnyInteger:$a, EncryptedIntegerType:$b);
    let results = (outs EncryptedIntegerType);

    let builders = [
        OpBuilder<(ins "Value":$a, "Value":$b), [{
         build($_builder, $_state, b.getType(), a, b);
        }]>
    ];

    let verifier = [{
        return ::mlir::concretelang::FHE::verifySubIntEintOp(*this);
    }];
}

def NegEintOp : FHE_Op<"neg_eint"> {

    let summary = "Negates an encrypted integer";

    let description = [{
        Negates an encrypted integer.
        The result must have the same width than the encrypted integer.

        Example:
        ```mlir
        // ok
        "FHE.neg_eint"(%a): (!FHE.eint<2>) -> (!FHE.eint<2>)

        // error
        "FHE.neg_eint"(%a): (!FHE.eint<2>) -> (!FHE.eint<3>)
        ```
    }];

    let arguments = (ins EncryptedIntegerType:$a);
    let results = (outs EncryptedIntegerType);

    let builders = [
        OpBuilder<(ins "Value":$a), [{
         build($_builder, $_state, a.getType(), a);
        }]>
    ];

    let verifier = [{
        return ::mlir::concretelang::FHE::verifyNegEintOp(*this);
    }];
}

def MulEintIntOp : FHE_Op<"mul_eint_int"> {

    let summary = "Mulitplies an encrypted integer and a clear integer";

    let description = [{
        Mulitplies an encrypted integer and a clear integer.
        The clear integer must have at most one more bit than the encrypted integer
        and the result must have the same width than the encrypted integer.

        Example:
        ```mlir
        // ok
        "FHE.mul_eint_int"(%a, %i) : (!FHE.eint<2>, i3) -> !FHE.eint<2>

        // error
        "FHE.mul_eint_int"(%a, %i) : (!FHE.eint<2>, i4) -> !FHE.eint<2>
        "FHE.mul_eint_int"(%a, %i) : (!FHE.eint<2>, i3) -> !FHE.eint<3>
        ```
    }];

    let arguments = (ins EncryptedIntegerType:$a, AnyInteger:$b);
    let results = (outs EncryptedIntegerType);

    let builders = [
        OpBuilder<(ins "Value":$a, "Value":$b), [{
         build($_builder, $_state, a.getType(), a, b);
        }]>
    ];

    let verifier = [{
        return ::mlir::concretelang::FHE::verifyMulEintIntOp(*this);
    }];
}

def ApplyLookupTableEintOp : FHE_Op<"apply_lookup_table"> {

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

    let arguments = (ins EncryptedIntegerType:$a,
        TensorOf<[AnyInteger]>:$lut);
    let results = (outs EncryptedIntegerType);

    let verifier = [{
        return ::mlir::concretelang::FHE::verifyApplyLookupTable(*this);
    }];
}

#endif
