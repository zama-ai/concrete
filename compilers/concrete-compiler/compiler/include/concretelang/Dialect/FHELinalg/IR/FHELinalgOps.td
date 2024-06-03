//===- FHELinalgOps.td - FHELinalg dialect ops ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONCRETELANG_DIALECT_FHELinalg_IR_FHELinalg_OPS
#define CONCRETELANG_DIALECT_FHELinalg_IR_FHELinalg_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

include "concretelang/Dialect/FHE/IR/FHEAttrs.td"
include "concretelang/Dialect/FHE/Interfaces/FHEInterfaces.td"

include "concretelang/Dialect/FHELinalg/IR/FHELinalgDialect.td"
include "concretelang/Dialect/FHELinalg/IR/FHELinalgTypes.td"

class FHELinalg_Op<string mnemonic, list<Trait> traits = []> :
    Op<FHELinalg_Dialect, mnemonic, traits>;

// TensorBroadcastingRules verify that the operands and result verify the broadcasting rules
def TensorBroadcastingRules : NativeOpTrait<"TensorBroadcastingRules">;
def TensorBinaryEintInt : NativeOpTrait<"TensorBinaryEintInt">;
def TensorBinaryIntEint : NativeOpTrait<"TensorBinaryIntEint">;
def TensorBinaryEint : NativeOpTrait<"TensorBinaryEint">;
def TensorUnaryEint : NativeOpTrait<"TensorUnaryEint">;


def FHELinalg_AddEintIntOp : FHELinalg_Op<"add_eint_int", [Pure, TensorBroadcastingRules, TensorBinaryEintInt, BinaryEintInt, DeclareOpInterfaceMethods<Binary>]> {
    let summary = "Returns a tensor that contains the addition of a tensor of encrypted integers and a tensor of clear integers.";

    let description = [{
        Performs an addition following the broadcasting rules between a tensor of encrypted integers and a tensor of clear integers.
        The width of the clear integers must be less than or equal to the width of encrypted integers.

        Examples:
        ```mlir
        // Returns the term-by-term addition of `%a0` with `%a1`
        "FHELinalg.add_eint_int"(%a0, %a1) : (tensor<4x!FHE.eint<4>>, tensor<4xi5>) -> tensor<4x!FHE.eint<4>>

        // Returns the term-by-term addition of `%a0` with `%a1`, where dimensions equal to one are stretched.
        "FHELinalg.add_eint_int"(%a0, %a1) : (tensor<4x1x4x!FHE.eint<4>>, tensor<1x4x4xi5>) -> tensor<4x4x4x!FHE.eint<4>>

        // Returns the addition of a 3x3 matrix of encrypted integers and a 3x1 matrix (a column) of integers.
        //
        // [1,2,3]   [1]   [2,3,4]
        // [4,5,6] + [2] = [6,7,8]
        // [7,8,9]   [3]   [10,11,12]
        //
        // The dimension #1 of operand #2 is stretched as it is equal to 1.
        "FHELinalg.add_eint_int"(%a0, %a1) : (tensor<3x3x!FHE.eint<4>>, tensor<3x1xi5>) -> tensor<3x3x!FHE.eint<4>>

        // Returns the addition of a 3x3 matrix of encrypted integers and a 1x3 matrix (a line) of integers.
        //
        // [1,2,3]             [2,4,6]
        // [4,5,6] + [1,2,3] = [5,7,9]
        // [7,8,9]             [8,10,12]
        //
        // The dimension #2 of operand #2 is stretched as it is equal to 1.
        "FHELinalg.add_eint_int"(%a0, %a1) : (tensor<3x3x!FHE.eint<4>>, tensor<1x3xi5>) -> tensor<3x3x!FHE.eint<4>>

        // Same behavior as the previous one, but as the dimension #2 is missing of operand #2.
        "FHELinalg.add_eint_int(%a0, %a1)" : (tensor<3x4x!FHE.eint<4>>, tensor<3xi5>) -> tensor<4x4x4x!FHE.eint<4>>

        ```
    }];

    let arguments = (ins
        Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>:$lhs,
        Type<And<[TensorOf<[AnyInteger]>.predicate, HasStaticShapePred]>>:$rhs
    );

    let results = (outs Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>);

    let builders = [
        OpBuilder<(ins "Value":$rhs, "Value":$lhs), [{
         build($_builder, $_state, rhs.getType(), rhs, lhs);
        }]>
    ];

    let hasFolder = 1;
}

def FHELinalg_AddEintOp : FHELinalg_Op<"add_eint", [Pure,
 TensorBroadcastingRules, TensorBinaryEint, BinaryEint, DeclareOpInterfaceMethods<BinaryEint>]> {
    let summary = "Returns a tensor that contains the addition of two tensor of encrypted integers.";

    let description = [{
        Performs an addition following the broadcasting rules between two tensors of encrypted integers.
        The width of the encrypted integers must be equal.

        Examples:
        ```mlir
        // Returns the term-by-term addition of `%a0` with `%a1`
        "FHELinalg.add_eint"(%a0, %a1) : (tensor<4x!FHE.eint<4>>, tensor<4x!FHE.eint<4>>) -> tensor<4x!FHE.eint<4>>

        // Returns the term-by-term addition of `%a0` with `%a1`, where dimensions equal to one are stretched.
        "FHELinalg.add_eint"(%a0, %a1) : (tensor<4x1x4x!FHE.eint<4>>, tensor<1x4x4x!FHE.eint<4>>) -> tensor<4x4x4x!FHE.eint<4>>

        // Returns the addition of a 3x3 matrix of encrypted integers and a 3x1 matrix (a column) of encrypted integers.
        //
        // [1,2,3]   [1]   [2,3,4]
        // [4,5,6] + [2] = [6,7,8]
        // [7,8,9]   [3]   [10,11,12]
        //
        // The dimension #1 of operand #2 is stretched as it is equal to 1.
        "FHELinalg.add_eint"(%a0, %a1) : (tensor<3x3x!FHE.eint<4>>, tensor<3x1x!FHE.eint<4>>) -> tensor<3x3x!FHE.eint<4>>

        // Returns the addition of a 3x3 matrix of encrypted integers and a 1x3 matrix (a line) of encrypted integers.
        //
        // [1,2,3]             [2,4,6]
        // [4,5,6] + [1,2,3] = [5,7,9]
        // [7,8,9]             [8,10,12]
        //
        // The dimension #2 of operand #2 is stretched as it is equal to 1.
        "FHELinalg.add_eint"(%a0, %a1) : (tensor<3x3x!FHE.eint<4>>, tensor<1x3x!FHE.eint<4>>) -> tensor<3x3x!FHE.eint<4>>

        // Same behavior as the previous one, but as the dimension #2 of operand #2 is missing.
        "FHELinalg.add_eint"(%a0, %a1) : (tensor<3x3x!FHE.eint<4>>, tensor<3x!FHE.eint<4>>) -> tensor<3x3x!FHE.eint<4>>
        ```
    }];

    let arguments = (ins
        Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>:$lhs,
        Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>:$rhs
    );

    let results = (outs Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>);

    let builders = [
        OpBuilder<(ins "Value":$rhs, "Value":$lhs), [{
         build($_builder, $_state, rhs.getType(), rhs, lhs);
        }]>
    ];
}

def FHELinalg_SubIntEintOp : FHELinalg_Op<"sub_int_eint", [Pure, TensorBroadcastingRules, TensorBinaryIntEint, BinaryIntEint, DeclareOpInterfaceMethods<Binary>]> {
    let summary = "Returns a tensor that contains the subtraction of a tensor of clear integers and a tensor of encrypted integers.";

    let description = [{
        Performs a subtraction following the broadcasting rules between a tensor of clear integers and a tensor of encrypted integers.
        The width of the clear integers must be less than or equal to the width of encrypted integers.

        Examples:
        ```mlir
        // Returns the term-by-term subtraction of `%a0` with `%a1`
        "FHELinalg.sub_int_eint"(%a0, %a1) : (tensor<4xi5>, tensor<4x!FHE.eint<4>>) -> tensor<4x!FHE.eint<4>>

        // Returns the term-by-term subtraction of `%a0` with `%a1`, where dimensions equal to one are stretched.
        "FHELinalg.sub_int_eint"(%a0, %a1) : (tensor<4x1x4xi5>, tensor<1x4x4x!FHE.eint<4>>) -> tensor<4x4x4x!FHE.eint<4>>

        // Returns the subtraction of a 3x3 matrix of integers and a 3x1 matrix (a column) of encrypted integers.
        //
        // [1,2,3]   [1]   [0,2,3]
        // [4,5,6] - [2] = [2,3,4]
        // [7,8,9]   [3]   [4,5,6]
        //
        // The dimension #1 of operand #2 is stretched as it is equal to 1.
        "FHELinalg.sub_int_eint"(%a0, %a1) : (tensor<3x3xi5>, tensor<3x1x!FHE.eint<4>>) -> tensor<3x3x!FHE.eint<4>>

        // Returns the subtraction of a 3x3 matrix of integers and a 1x3 matrix (a line) of encrypted integers.
        //
        // [1,2,3]             [0,0,0]
        // [4,5,6] - [1,2,3] = [3,3,3]
        // [7,8,9]             [6,6,6]
        //
        // The dimension #2 of operand #2 is stretched as it is equal to 1.
        "FHELinalg.sub_int_eint"(%a0, %a1) : (tensor<3x3xi5>, tensor<1x3x!FHE.eint<4>>) -> tensor<3x3x!FHE.eint<4>>

        // Same behavior as the previous one, but as the dimension #2 is missing of operand #2.
        "FHELinalg.sub_int_eint"(%a0, %a1) : (tensor<3x3xi5>, tensor<3x!FHE.eint<4>>) -> tensor<3x3x!FHE.eint<4>>

        ```
    }];

    let arguments = (ins
        Type<And<[TensorOf<[AnyInteger]>.predicate, HasStaticShapePred]>>:$lhs,
        Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>:$rhs
    );

    let results = (outs Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>);

    let builders = [
        OpBuilder<(ins "Value":$rhs, "Value":$lhs), [{
         build($_builder, $_state, lhs.getType(), rhs, lhs);
        }]>
    ];
}

def FHELinalg_SubEintIntOp : FHELinalg_Op<"sub_eint_int", [Pure, TensorBroadcastingRules, TensorBinaryEintInt, BinaryEintInt, DeclareOpInterfaceMethods<Binary>]> {
    let summary = "Returns a tensor that contains the subtraction of a tensor of clear integers from a tensor of encrypted integers.";

    let description = [{
        Performs a subtraction following the broadcasting rules between a tensor of clear integers from a tensor of encrypted integers.
        The width of the clear integers must be less than or equal to the width of encrypted integers.

        Examples:
        ```mlir
        // Returns the term-by-term subtraction of `%a0` with `%a1`
        "FHELinalg.sub_eint_int"(%a0, %a1) : (tensor<4x!FHE.eint<4>>, tensor<4xi5>) -> tensor<4x!FHE.eint<4>>

        // Returns the term-by-term subtraction of `%a0` with `%a1`, where dimensions equal to one are stretched.
        "FHELinalg.sub_eint_int"(%a0, %a1) : (tensor<1x4x4x!FHE.eint<4>>, tensor<4x1x4xi5>) -> tensor<4x4x4x!FHE.eint<4>>

        // Returns the subtraction of a 3x3 matrix of integers and a 3x1 matrix (a column) of encrypted integers.
        //
        // [1,2,3]   [1]   [0,2,3]
        // [4,5,6] - [2] = [2,3,4]
        // [7,8,9]   [3]   [4,5,6]
        //
        // The dimension #1 of operand #2 is stretched as it is equal to 1.
        "FHELinalg.sub_eint_int"(%a0, %a1) : (tensor<3x1x!FHE.eint<4>>, tensor<3x3xi5>) -> tensor<3x3x!FHE.eint<4>>

        // Returns the subtraction of a 3x3 matrix of integers and a 1x3 matrix (a line) of encrypted integers.
        //
        // [1,2,3]             [0,0,0]
        // [4,5,6] - [1,2,3] = [3,3,3]
        // [7,8,9]             [6,6,6]
        //
        // The dimension #2 of operand #2 is stretched as it is equal to 1.
        "FHELinalg.sub_eint_int"(%a0, %a1) : (tensor<1x3x!FHE.eint<4>>, tensor<3x3xi5>) -> tensor<3x3x!FHE.eint<4>>

        // Same behavior as the previous one, but as the dimension #2 is missing of operand #2.
        "FHELinalg.sub_eint_int"(%a0, %a1) : (tensor<3x!FHE.eint<4>>, tensor<3x3xi5>) -> tensor<3x3x!FHE.eint<4>>

        ```
    }];

    let arguments = (ins
        Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>:$lhs,
        Type<And<[TensorOf<[AnyInteger]>.predicate, HasStaticShapePred]>>:$rhs
    );

    let results = (outs Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>);

    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs), [{
            build($_builder, $_state, lhs.getType(), lhs, rhs);
        }]>

    ];

    let hasFolder = 1;
}

def FHELinalg_SubEintOp : FHELinalg_Op<"sub_eint", [Pure, TensorBroadcastingRules, TensorBinaryEint, BinaryEint, DeclareOpInterfaceMethods<BinaryEint>]> {
    let summary = "Returns a tensor that contains the subtraction of two tensor of encrypted integers.";

    let description = [{
        Performs an subtraction following the broadcasting rules between two tensors of encrypted integers.
        The width of the encrypted integers must be equal.

        Examples:
        ```mlir
        // Returns the term-by-term subtraction of `%a0` with `%a1`
        "FHELinalg.sub_eint"(%a0, %a1) : (tensor<4x!FHE.eint<4>>, tensor<4x!FHE.eint<4>>) -> tensor<4x!FHE.eint<4>>

        // Returns the term-by-term subtraction of `%a0` with `%a1`, where dimensions equal to one are stretched.
        "FHELinalg.sub_eint"(%a0, %a1) : (tensor<4x1x4x!FHE.eint<4>>, tensor<1x4x4x!FHE.eint<4>>) -> tensor<4x4x4x!FHE.eint<4>>

        // Returns the subtraction of a 3x3 matrix of integers and a 3x1 matrix (a column) of encrypted integers.
        //
        // [1,2,3]   [1]   [0,2,3]
        // [4,5,6] - [2] = [2,3,4]
        // [7,8,9]   [3]   [4,5,6]
        //
        // The dimension #1 of operand #2 is stretched as it is equal to 1.
        "FHELinalg.sub_eint"(%a0, %a1) : (tensor<3x3x!FHE.eint<4>>, tensor<3x1x!FHE.eint<4>>) -> tensor<3x3x!FHE.eint<4>>

        // Returns the subtraction of a 3x3 matrix of integers and a 1x3 matrix (a line) of encrypted integers.
        //
        // [1,2,3]             [0,0,0]
        // [4,5,6] - [1,2,3] = [3,3,3]
        // [7,8,9]             [6,6,6]
        //
        // The dimension #2 of operand #2 is stretched as it is equal to 1.
        "FHELinalg.sub_eint"(%a0, %a1) : (tensor<3x3x!FHE.eint<4>>, tensor<1x3x!FHE.eint<4>>) -> tensor<3x3x!FHE.eint<4>>

        // Same behavior as the previous one, but as the dimension #2 of operand #2 is missing.
        "FHELinalg.sub_eint"(%a0, %a1) : (tensor<3x3x!FHE.eint<4>>, tensor<3x!FHE.eint<4>>) -> tensor<3x3x!FHE.eint<4>>
        ```
    }];

    let arguments = (ins
        Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>:$lhs,
        Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>:$rhs
    );

    let results = (outs Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>);

    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs), [{
            build($_builder, $_state, lhs.getType(), lhs, rhs);
        }]>
    ];
}

def FHELinalg_NegEintOp : FHELinalg_Op<"neg_eint", [Pure, TensorUnaryEint, UnaryEint, DeclareOpInterfaceMethods<UnaryEint>]> {
    let summary = "Returns a tensor that contains the negation of a tensor of encrypted integers.";

    let description = [{
        Performs a negation to a tensor of encrypted integers.

        Examples:
        ```mlir
        // Returns the term-by-term negation of `%a0`
        "FHELinalg.neg_eint"(%a0) : (tensor<3x3x!FHE.eint<4>>) -> tensor<3x3x!FHE.eint<4>>
        //
        //        ( [1,2,3] )   [31,30,29]
        // negate ( [4,5,6] ) = [28,27,26]
        //        ( [7,8,9] )   [25,24,23]
        //
        // The negation is computed as `2**(p+1) - a` where p=4 here.
        ```
    }];

    let arguments = (ins
        Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>:$input
    );

    let results = (outs Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>);

    let builders = [
        OpBuilder<(ins "Value":$tensor), [{
         build($_builder, $_state, tensor.getType(), tensor);
        }]>
    ];
}

def FHELinalg_MulEintIntOp : FHELinalg_Op<"mul_eint_int", [Pure, TensorBroadcastingRules, TensorBinaryEintInt, BinaryEintInt, DeclareOpInterfaceMethods<Binary, ["sqMANP"]>]> {
    let summary = "Returns a tensor that contains the multiplication of a tensor of encrypted integers and a tensor of clear integers.";

    let description = [{
        Performs a multiplication following the broadcasting rules between a tensor of encrypted integers and a tensor of clear integers.
        The width of the clear integers must be less than or equal to the width of encrypted integers.

        Examples:
        ```mlir
        // Returns the term-by-term multiplication of `%a0` with `%a1`
        "FHELinalg.mul_eint_int"(%a0, %a1) : (tensor<4x!FHE.eint<4>>, tensor<4xi5>) -> tensor<4x!FHE.eint<4>>

        // Returns the term-by-term multiplication of `%a0` with `%a1`, where dimensions equal to one are stretched.
        "FHELinalg.mul_eint_int"(%a0, %a1) : (tensor<4x1x4x!FHE.eint<4>>, tensor<1x4x4xi5>) -> tensor<4x4x4x!FHE.eint<4>>

        // Returns the multiplication of a 3x3 matrix of encrypted integers and a 3x1 matrix (a column) of integers.
        //
        // [1,2,3]   [1]   [1,2,3]
        // [4,5,6] * [2] = [8,10,18]
        // [7,8,9]   [3]   [21,24,27]
        //
        // The dimension #1 of operand #2 is stretched as it is equal to 1.
        "FHELinalg.mul_eint_int"(%a0, %a1) : (tensor<3x3x!FHE.eint<4>>, tensor<3x1xi5>) -> tensor<3x3x!FHE.eint<4>>

        // Returns the multiplication of a 3x3 matrix of encrypted integers and a 1x3 matrix (a line) of integers.
        //
        // [1,2,3]             [2,4,6]
        // [4,5,6] * [1,2,3] = [5,7,9]
        // [7,8,9]             [8,10,12]
        //
        // The dimension #2 of operand #2 is stretched as it is equal to 1.
        "FHELinalg.mul_eint_int"(%a0, %a1) : (tensor<3x3x!FHE.eint<4>>, tensor<1x3xi5>) -> tensor<3x3x!FHE.eint<4>>

        // Same behavior as the previous one, but as the dimension #2 is missing of operand #2.
        "FHELinalg.mul_eint_int"(%a0, %a1) : (tensor<3x3x!FHE.eint<4>>, tensor<3xi5>) -> tensor<3x3x!FHE.eint<4>>

        ```
    }];

    let arguments = (ins
        Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>:$lhs,
        Type<And<[TensorOf<[AnyInteger]>.predicate, HasStaticShapePred]>>:$rhs
    );

    let results = (outs Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>);

    let hasFolder = 1;

    let hasCanonicalizer = 1;
}

def FHELinalg_MulEintOp : FHELinalg_Op<"mul_eint", [Pure, TensorBroadcastingRules, TensorBinaryEint, BinaryEint, DeclareOpInterfaceMethods<BinaryEint>]> {
    let summary = "Returns a tensor that contains the multiplication of two tensor of encrypted integers.";

    let description = [{
        Performs an addition following the broadcasting rules between two tensors of encrypted integers.
        The width of the encrypted integers must be equal.

        Examples:
        ```mlir
        // Returns the term-by-term multiplication of `%a0` with `%a1`
        "FHELinalg.mul_eint"(%a0, %a1) : (tensor<4x!FHE.eint<8>>, tensor<4x!FHE.eint<8>>) -> tensor<4x!FHE.eint<8>>

        // Returns the term-by-term multiplication of `%a0` with `%a1`, where dimensions equal to one are stretched.
        "FHELinalg.mul_eint"(%a0, %a1) : (tensor<4x1x4x!FHE.eint<8>>, tensor<1x4x4x!FHE.eint<8>>) -> tensor<4x4x4x!FHE.eint<8>>

        // Returns the multiplication of a 3x3 matrix of encrypted integers and a 3x1 matrix (a column) of encrypted integers.
        //
        // [1,2,3]   [1]   [1,2,3]
        // [4,5,6] * [2] = [8,10,12]
        // [7,8,9]   [3]   [21,24,27]
        //
        // The dimension #1 of operand #2 is stretched as it is equal to 1.
        "FHELinalg.mul_eint"(%a0, %a1) : (tensor<3x3x!FHE.eint<8>>, tensor<3x1x!FHE.eint<8>>) -> tensor<3x3x!FHE.eint<8>>

        // Returns the multiplication of a 3x3 matrix of encrypted integers and a 1x3 matrix (a line) of encrypted integers.
        //
        // [1,2,3]             [1,4,9]
        // [4,5,6] * [1,2,3] = [4,10,18]
        // [7,8,9]             [7,16,27]
        //
        // The dimension #2 of operand #2 is stretched as it is equal to 1.
        "FHELinalg.mul_eint"(%a0, %a1) : (tensor<3x3x!FHE.eint<8>>, tensor<1x3x!FHE.eint<8>>) -> tensor<3x3x!FHE.eint<8>>

        // Same behavior as the previous one, but as the dimension #2 of operand #2 is missing.
        "FHELinalg.mul_eint"(%a0, %a1) : (tensor<3x3x!FHE.eint<8>>, tensor<3x!FHE.eint<8>>) -> tensor<3x3x!FHE.eint<8>>
        ```
    }];

    let arguments = (ins
        Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>:$lhs,
        Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>:$rhs
    );

    let results = (outs Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>);

    let builders = [
        OpBuilder<(ins "Value":$rhs, "Value":$lhs), [{
         build($_builder, $_state, rhs.getType(), rhs, lhs);
        }]>
    ];
}

def FHELinalg_ApplyLookupTableEintOp : FHELinalg_Op<"apply_lookup_table", [Pure, ConstantNoise]> {
    let summary = "Returns a tensor that contains the result of the lookup on a table.";

    let description = [{
        For each encrypted index, performs a lookup table of clear integers.

        ```mlir
        // The result of this operation, is a tensor that contains the result of a lookup table.
        // i.e. %res[i, ..., k] = %lut[%t[i, ..., k]]
        %res = FHELinalg.apply_lookup_table(%t, %lut): tensor<DNx...xD1x!FHE.eint<$p>>, tensor<D2^$pxi64> -> tensor<DNx...xD1x!FHE.eint<$p>>
        ```

        The `%lut` argument must be a tensor with one dimension, where its dimension is `2^p` where `p` is the width of the encrypted integers.

        Examples:
        ```mlir

        // Returns the lookup of 3x3 matrix of encrypted indices of with 2 on a table of size 4=2² of clear integers.
        //
        // [0,1,2]                 [1,3,5]
        // [3,0,1] lut [1,3,5,7] = [7,1,3]
        // [2,3,0]                 [5,7,1]
        "FHELinalg.apply_lookup_table"(%t, %lut) : (tensor<3x3x!FHE.eint<2>>, tensor<4xi64>) -> tensor<3x3x!FHE.eint<3>>
        ```
    }];

    let arguments = (ins
        Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>:$t,
        Type<And<[TensorOf<[AnyInteger]>.predicate, HasStaticShapePred]>>:$lut
    );

    let results = (outs Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>);

    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

def FHELinalg_ApplyMultiLookupTableEintOp : FHELinalg_Op<"apply_multi_lookup_table", [Pure, ConstantNoise]> {
    let summary = "Returns a tensor that contains the result of the lookup on a table, using a different lookup table for each element.";

    let description = [{
        Performs for each encrypted index a lookup table of clear integers. Multiple lookup tables are passed, and the application of lookup tables
        is performed following the broadcasting rules.

        ```mlir
        // The result of this operation, is a tensor that contains the result of the lookup on different tables.
        // i.e. %res[i, ..., k] = [ %luts[i][%t[i]], ..., %luts[k][%t[k]] ]
        %res = FHELinalg.apply_multi_lookup_table(%t, %lut): tensor<DNx...xD1x!FHE.eint<$p>>, tensor<DMx...xD1xD2^$pxi64> -> tensor<DNx...xD1x!FHE.eint<$p>>
        ```

        The `%luts` argument should be a tensor with M dimension, where the first M-1 dimensions are broadcastable with the N dimensions of the encrypted tensor,
        and where the last dimension dimension is equal to `2^p` where `p` is the width of the encrypted integers.

        Examples:
        ```mlir

        // Returns the lookup of 3x2 matrix of encrypted indices of width 2 on a vector of 2 tables of size 4=2² of clear integers.
        // The tables are broadcasted along the first dimension of the tensor.
        //
        // [0,1]                            = [1,2]
        // [3,0] lut [[1,3,5,7], [0,2,4,6]] = [7,0]
        // [2,3]                            = [5,6]
        "FHELinalg.apply_multi_lookup_table"(%t, %luts) : (tensor<3x2x!FHE.eint<2>>, tensor<2x4xi64>) -> tensor<3x2x!FHE.eint<3>>
        ```

        ```mlir

        // Returns the lookup of a vector of 3 encrypted indices of width 2 on a vector of 3 tables of size 4=2² of clear integers.
        //
        // [3,0,1] lut [[1,3,5,7], [0,2,4,6], [1,2,3,4]] = [7,0,2]
        "FHELinalg.apply_multi_lookup_table"(%t, %luts) : (tensor<3x!FHE.eint<2>>, tensor<3x4xi64>) -> tensor<3x!FHE.eint<3>>
        ```
    }];

    let arguments = (ins
        Type<And<[TensorOf<[FHE_EncryptedUnsignedIntegerType]>.predicate, HasStaticShapePred]>>:$t,
        Type<And<[TensorOf<[AnyInteger]>.predicate, HasStaticShapePred]>>:$luts
    );

    let results = (outs Type<And<[TensorOf<[FHE_EncryptedUnsignedIntegerType]>.predicate, HasStaticShapePred]>>);

    let hasVerifier = 1;
}

def FHELinalg_ApplyMappedLookupTableEintOp : FHELinalg_Op<"apply_mapped_lookup_table", [Pure, ConstantNoise]> {
    let summary = "Returns a tensor that contains the result of the lookup on a table, using a different lookup table for each element, specified by a map.";

    let description = [{
        Performs for each encrypted index a lookup table of clear integers. Multiple lookup tables are passed, and the application of lookup tables
        is performed following the broadcasting rules. The precise lookup is specified by a map.

        ```mlir
        // The result of this operation, is a tensor that contains the result of the lookup on different tables.
        // i.e. %res[i, ..., k] = %luts[ %map[i, ..., k] ][ %t[i, ..., k] ]
        %res = FHELinalg.apply_mapped_lookup_table(%t, %luts, %map): tensor<DNx...xD1x!FHE.eint<$p>>, tensor<DM x ^$p>, tensor<DNx...xD1xindex> -> tensor<DNx...xD1x!FHE.eint<$p>>
        ```

        Examples:
        ```mlir

        // Returns the lookup of 3x2 matrix of encrypted indices of width 2 on a vector of 2 tables of size 4=2^2 of clear integers.
        //
        // [0,1]                                 [0, 1] = [1,2]
        // [3,0] lut [[1,3,5,7], [0,2,4,6]] with [0, 1] = [7,0]
        // [2,3]                                 [0, 1] = [5,6]
        "FHELinalg.apply_mapped_lookup_table"(%t, %luts, %map) : (tensor<3x2x!FHE.eint<2>>, tensor<2x4xi64>, tensor<3x2xindex>) -> tensor<3x2x!FHE.eint<3>>
        ```

        Others examples:
        // [0,1]                                 [1, 0] = [3,2]
        // [3,0] lut [[1,3,5,7], [0,2,4,6]] with [0, 1] = [7,0]
        // [2,3]                                 [1, 0] = [4,7]

        // [0,1]                                 [0, 0] = [1,3]
        // [3,0] lut [[1,3,5,7], [0,2,4,6]] with [1, 1] = [6,0]
        // [2,3]                                 [1, 0] = [4,7]

        // [0,1]                                 [0]    = [1,3]
        // [3,0] lut [[1,3,5,7], [0,2,4,6]] with [1]    = [6,0]
        // [2,3]                                 [0]    = [5,7]

        // [0,1]                                        = [1,2]
        // [3,0] lut [[1,3,5,7], [0,2,4,6]] with [0, 1] = [7,0]
        // [2,3]                                        = [5,6]

    }];

    let arguments = (ins
        Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>:$t,
        Type<And<[TensorOf<[AnyInteger]>.predicate, HasStaticShapePred]>>:$luts,
        Type<And<[TensorOf<[Index]>.predicate, HasStaticShapePred]>>:$map
    );

    let results = (outs Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>);

    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

def FHELinalg_Dot : FHELinalg_Op<"dot_eint_int", [Pure, BinaryEintInt, DeclareOpInterfaceMethods<Binary, ["sqMANP"]>]> {
    let summary = "Returns the encrypted dot product between a vector of encrypted integers and a vector of clean integers.";

    let description = [{
        Performs a dot product between a vector of encrypted integers and a vector of clear integers.

        Examples:
        ```mlir
        // Returns the dot product of `%a0` with `%a1`
        "FHELinalg.dot_eint_int"(%a0, %a1) : (tensor<4x!FHE.eint<4>>, tensor<4xi5>) -> !FHE.eint<4>

        ```
    }];

    let arguments = (ins
        Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred, HasAnyRankOfPred<[1]>]>>:$lhs,
        Type<And<[TensorOf<[AnyInteger]>.predicate, HasStaticShapePred, HasAnyRankOfPred<[1]>]>>:$rhs);

    let results = (outs FHE_AnyEncryptedInteger:$out);

    let hasVerifier = 1;
}

def FHELinalg_DotEint : FHELinalg_Op<"dot_eint_eint", [Pure]> {
    let summary = "Returns the encrypted dot product between two vectors of encrypted integers.";

    let description = [{
        Performs a dot product between two vectors of encrypted integers.

        Examples:
        ```mlir
        // Returns the dot product of `%a0` with `%a1`
        "FHELinalg.dot_eint_eint"(%a0, %a1) : (tensor<4x!FHE.eint<4>>, tensor<4x!FHE.eint<4>>) -> !FHE.eint<4>

        ```
    }];

    let arguments = (ins
        Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred, HasAnyRankOfPred<[1]>]>>:$lhs,
        Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred, HasAnyRankOfPred<[1]>]>>:$rhs);

    let results = (outs FHE_AnyEncryptedInteger:$out);

    let hasVerifier = 1;
}

def FHELinalg_MatMulEintIntOp : FHELinalg_Op<"matmul_eint_int", [Pure, TensorBinaryEintInt, BinaryEintInt, DeclareOpInterfaceMethods<Binary, ["sqMANP"]>]> {
    let summary = "Returns a tensor that contains the result of the matrix multiplication of a matrix of encrypted integers and a matrix of clear integers.";

    let description = [{
        Performs a matrix multiplication of a matrix of encrypted integers and a matrix of clear integers.
        The width of the clear integers must be less than or equal to the width of encrypted integers.

        The behavior depends on the arguments in the following way:

            - If both arguments are 2-D,
              they are multiplied like conventional matrices.

              e.g.,

              arg0: tensor<MxN> = [...]
              arg1: tensor<NxP> = [...]

              result: tensor<MxP> = [...]

            - If the first argument is a vector (1-D),
              it is treated as a matrix with a single row and standard matrix multiplication is performed.

              After standard matrix multiplication,
              the first dimension is removed from the result.

              e.g.,

              arg0: tensor<3> = [x, y, z]
              arg1: tensor<3xM> = [
                  [_, _, ..., _, _],
                  [_, _, ..., _, _],
                  [_, _, ..., _, _],
              ]

              is treated as

              arg0: tensor<1x3> = [
                  [x, y, z]
              ]
              arg1: tensor<3xM> = [
                  [_, _, ..., _, _],
                  [_, _, ..., _, _],
                  [_, _, ..., _, _],
              ]

              and matrix multiplication is performed with the following form (1x3 @ 3xM -> 1xM)

              result: tensor<1xM> = [[_, _, ..., _, _]]

              finally, the first dimension is removed by definition so the result has the following form

              result: tensor<M>  = [_, _, ..., _, _]

            - If the second argument is 1-D,
              it is treated as a matrix with a single column and standard matrix multiplication is performed.

              After standard matrix multiplication,
              the last dimension is removed from the result.

              e.g.,

              arg0: tensor<Mx3> = [
                  [_, _, _],
                  [_, _, _],
                  ...,
                  [_, _, _],
                  [_, _, _],
              ]
              arg1: tensor<3> = [x, y, z]

              is treated as

              arg0: tensor<Mx3> = [
                  [_, _, _],
                  [_, _, _],
                  ...,
                  [_, _, _],
                  [_, _, _],
              ]
              arg1: tensor<3x1> = [
                [x],
                [y],
                [z],
              ]

              and matrix multiplication is performed with the following form (Mx3 @ 3x1 -> Mx1)

              result: tensor<Mx1> = [
                [_],
                [_],
                  ...,
                [_],
                [_],
              ]

              finally, the last dimension is removed by definition so the result has the following form

              result: tensor<M> = [_, _, _]

            - If either argument is N-D where N > 2,
              the operation is treated as a collection of matrices residing in the last two indices and broadcasted accordingly.

              arg0: tensor<Kx1MxN> = [...]
              arg1: tensor<LxNxP> = [...]

              result: tensor<KxLxMxP> = [...]

        ```mlir
        "FHELinalg.matmul_eint_int(%a, %b) : (tensor<MxNx!FHE.eint<p>>, tensor<NxPxip'>) -> tensor<MxPx!FHE.eint<p>>"
        "FHELinalg.matmul_eint_int(%a, %b) : (tensor<KxLxMxNx!FHE.eint<p>>, tensor<KxLxNxPxip'>) -> tensor<KxLxMxPx!FHE.eint<p>>"
        "FHELinalg.matmul_eint_int(%a, %b) : (tensor<MxNx!FHE.eint<p>>, tensor<Nxip'>) -> tensor<Mx!FHE.eint<p>>"
        "FHELinalg.matmul_eint_int(%a, %b) : (tensor<Nx!FHE.eint<p>>, tensor<NxPxip'>) -> tensor<Px!FHE.eint<p>>"
        ```

        Examples:
        ```mlir
        // Returns the matrix multiplication of a 3x2 matrix of encrypted integers and a 2x3 matrix of integers.
        //         [ 1, 2, 3]
        //         [ 2, 3, 4]
        //       *
        // [1,2]   [ 5, 8,11]
        // [3,4] = [11,18,25]
        // [5,6]   [17,28,39]
        //
        "FHELinalg.matmul_eint_int"(%a, %b) : (tensor<3x2x!FHE.eint<6>>, tensor<2x3xi7>) -> tensor<3x3x!FHE.eint<6>>
        ```
    }];

    let arguments = (ins
        Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>:$lhs,
        Type<And<[TensorOf<[AnyInteger]>.predicate, HasStaticShapePred]>>:$rhs
    );

    let results = (outs Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>);

    let hasVerifier = 1;

    let hasCanonicalizer = 1;

    let extraClassDeclaration = [{
        mlir::Value getClearMatrix() { return getRhs(); }
        mlir::Value getEncryptedMatrix () { return getLhs(); }
    }];
}

def FHELinalg_MatMulIntEintOp : FHELinalg_Op<"matmul_int_eint", [Pure, TensorBinaryIntEint, BinaryIntEint, DeclareOpInterfaceMethods<Binary, ["sqMANP"]>]> {
    let summary = "Returns a tensor that contains the result of the matrix multiplication of a matrix of clear integers and a matrix of encrypted integers.";

    let description = [{
        Performs a matrix multiplication of a matrix of clear integers and a matrix of encrypted integers.
        The width of the clear integers must be less than or equal to the width of encrypted integers.

        The behavior depends on the arguments in the following way:

            - If both arguments are 2-D,
              they are multiplied like conventional matrices.

              e.g.,

              arg0: tensor<MxN> = [...]
              arg1: tensor<NxP> = [...]

              result: tensor<MxP> = [...]

            - If the first argument is a vector (1-D),
              it is treated as a matrix with a single row and standard matrix multiplication is performed.

              After standard matrix multiplication,
              the first dimension is removed from the result.

              e.g.,

              arg0: tensor<3> = [x, y, z]
              arg1: tensor<3xM> = [
                  [_, _, ..., _, _],
                  [_, _, ..., _, _],
                  [_, _, ..., _, _],
              ]

              is treated as

              arg0: tensor<1x3> = [
                  [x, y, z]
              ]
              arg1: tensor<3xM> = [
                  [_, _, ..., _, _],
                  [_, _, ..., _, _],
                  [_, _, ..., _, _],
              ]

              and matrix multiplication is performed with the following form (1x3 @ 3xM -> 1xM)

              result: tensor<1xM> = [[_, _, ..., _, _]]

              finally, the first dimension is removed by definition so the result has the following form

              result: tensor<M>  = [_, _, ..., _, _]

            - If the second argument is 1-D,
              it is treated as a matrix with a single column and standard matrix multiplication is performed.

              After standard matrix multiplication,
              the last dimension is removed from the result.

              e.g.,

              arg0: tensor<Mx3> = [
                  [_, _, _],
                  [_, _, _],
                  ...,
                  [_, _, _],
                  [_, _, _],
              ]
              arg1: tensor<3> = [x, y, z]

              is treated as

              arg0: tensor<Mx3> = [
                  [_, _, _],
                  [_, _, _],
                  ...,
                  [_, _, _],
                  [_, _, _],
              ]
              arg1: tensor<3x1> = [
                [x],
                [y],
                [z],
              ]

              and matrix multiplication is performed with the following form (Mx3 @ 3x1 -> Mx1)

              result: tensor<Mx1> = [
                [_],
                [_],
                  ...,
                [_],
                [_],
              ]

              finally, the last dimension is removed by definition so the result has the following form

              result: tensor<M> = [_, _, _]

            - If either argument is N-D where N > 2,
              the operation is treated as a collection of matrices residing in the last two indices and broadcasted accordingly.

              arg0: tensor<Kx1MxN> = [...]
              arg1: tensor<LxNxP> = [...]

              result: tensor<KxLxMxP> = [...]

        ```mlir
        "FHELinalg.matmul_int_eint(%a, %b) : (tensor<MxNxip'>, tensor<NxPxFHE.eint<p>>) -> tensor<MxPx!FHE.eint<p>>"
        "FHELinalg.matmul_int_eint(%a, %b) : (tensor<KxLxMxNxip'>, tensor<KxLxNxPxFHE.eint<p>>) -> tensor<KxLxMxPx!FHE.eint<p>>"
        "FHELinalg.matmul_int_eint(%a, %b) : (tensor<MxNxip'>, tensor<NxFHE.eint<p>>) -> tensor<Mx!FHE.eint<p>>"
        "FHELinalg.matmul_int_eint(%a, %b) : (tensor<Nxip'>, tensor<NxPxFHE.eint<p>>) -> tensor<Px!FHE.eint<p>>"
        ```

        Examples:
        ```mlir
        // Returns the matrix multiplication of a 3x2 matrix of clear integers and a 2x3 matrix of encrypted integers.
        //         [ 1, 2, 3]
        //         [ 2, 3, 4]
        //       *
        // [1,2]   [ 5, 8,11]
        // [3,4] = [11,18,25]
        // [5,6]   [17,28,39]
        //
        "FHELinalg.matmul_int_eint"(%a, %b) : (tensor<3x2xi7>, tensor<2x3x!FHE.eint<6>>) -> tensor<3x3x!FHE.eint<6>>
        ```
    }];

    let arguments = (ins
        Type<And<[TensorOf<[AnyInteger]>.predicate, HasStaticShapePred]>>:$lhs,
        Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>:$rhs
    );

    let results = (outs Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>);

    let hasVerifier = 1;

    let hasCanonicalizer = 1;

    let extraClassDeclaration = [{
        mlir::Value getClearMatrix() { return getLhs(); }
        mlir::Value getEncryptedMatrix () { return getRhs(); }
    }];
}


def FHELinalg_MatMulEintEintOp : FHELinalg_Op<"matmul_eint_eint", [Pure, TensorBinaryEint]> {
    let summary = "Returns a tensor that contains the result of the matrix multiplication of a matrix of encrypted integers and a second matrix of encrypted integers.";

    let description = [{
        Performs a matrix multiplication of a matrix of encrypted integers and a second matrix of encrypted integers.

        The behavior depends on the arguments in the following way:

            - If both arguments are 2-D,
              they are multiplied like conventional matrices.

              e.g.,

              arg0: tensor<MxN> = [...]
              arg1: tensor<NxP> = [...]

              result: tensor<MxP> = [...]

            - If the first argument is a vector (1-D),
              it is treated as a matrix with a single row and standard matrix multiplication is performed.

              After standard matrix multiplication,
              the first dimension is removed from the result.

              e.g.,

              arg0: tensor<3> = [x, y, z]
              arg1: tensor<3xM> = [
                  [_, _, ..., _, _],
                  [_, _, ..., _, _],
                  [_, _, ..., _, _],
              ]

              is treated as

              arg0: tensor<1x3> = [
                  [x, y, z]
              ]
              arg1: tensor<3xM> = [
                  [_, _, ..., _, _],
                  [_, _, ..., _, _],
                  [_, _, ..., _, _],
              ]

              and matrix multiplication is performed with the following form (1x3 @ 3xM -> 1xM)

              result: tensor<1xM> = [[_, _, ..., _, _]]

              finally, the first dimension is removed by definition so the result has the following form

              result: tensor<M>  = [_, _, ..., _, _]

            - If the second argument is 1-D,
              it is treated as a matrix with a single column and standard matrix multiplication is performed.

              After standard matrix multiplication,
              the last dimension is removed from the result.

              e.g.,

              arg0: tensor<Mx3> = [
                  [_, _, _],
                  [_, _, _],
                  ...,
                  [_, _, _],
                  [_, _, _],
              ]
              arg1: tensor<3> = [x, y, z]

              is treated as

              arg0: tensor<Mx3> = [
                  [_, _, _],
                  [_, _, _],
                  ...,
                  [_, _, _],
                  [_, _, _],
              ]
              arg1: tensor<3x1> = [
                [x],
                [y],
                [z],
              ]

              and matrix multiplication is performed with the following form (Mx3 @ 3x1 -> Mx1)

              result: tensor<Mx1> = [
                [_],
                [_],
                  ...,
                [_],
                [_],
              ]

              finally, the last dimension is removed by definition so the result has the following form

              result: tensor<M> = [_, _, _]

            - If either argument is N-D where N > 2,
              the operation is treated as a collection of matrices residing in the last two indices and broadcasted accordingly.

              arg0: tensor<Kx1MxN> = [...]
              arg1: tensor<LxNxP> = [...]

              result: tensor<KxLxMxP> = [...]

        ```mlir
        "FHELinalg.matmul_eint_eint(%a, %b) : (tensor<MxNx!FHE.eint<p>>, tensor<NxPx!FHE.eint<p>'>) -> tensor<MxPx!FHE.eint<p>>"
        "FHELinalg.matmul_eint_eint(%a, %b) : (tensor<KxLxMxNx!FHE.eint<p>>, tensor<KxLxNxPx!FHE.eint<p>'>) -> tensor<KxLxMxPx!FHE.eint<p>>"
        "FHELinalg.matmul_eint_eint(%a, %b) : (tensor<MxNx!FHE.eint<p>>, tensor<Nx!FHE.eint<p>'>) -> tensor<Mx!FHE.eint<p>>"
        "FHELinalg.matmul_eint_eint(%a, %b) : (tensor<Nx!FHE.eint<p>>, tensor<NxPx!FHE.eint<p>'>) -> tensor<Px!FHE.eint<p>>"
        ```

        Examples:
        ```mlir
        // Returns the matrix multiplication of a 3x2 matrix of encrypted integers and a 2x3 matrix of integers.
        //         [ 1, 2, 3]
        //         [ 2, 3, 4]
        //       *
        // [1,2]   [ 5, 8,11]
        // [3,4] = [11,18,25]
        // [5,6]   [17,28,39]
        //
        "FHELinalg.matmul_eint_eint"(%a, %b) : (tensor<3x2x!FHE.eint<6>>, tensor<2x3x!FHE.eint<6>>) -> tensor<3x3x!FHE.eint<12>>
        ```
    }];

    let arguments = (ins
        Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>:$lhs,
        Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>:$rhs
    );

    let results = (outs Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>);

    let hasVerifier = 1;
}

def FHELinalg_SumOp : FHELinalg_Op<"sum", [Pure, TensorUnaryEint]> {
    let summary = "Returns the sum of elements of a tensor of encrypted integers along specified axes.";

    let description = [{
        Attributes:

        - keep_dims: boolean = false
            whether to keep the rank of the tensor after the sum operation
            if true, reduced axes will have the size of 1

        - axes: I64ArrayAttr = []
            list of dimension to perform the sum along
            think of it as the dimensions to reduce (see examples below to get an intuition)

        Examples:

        ```mlir
        // Returns the sum of all elements of `%a0`
        "FHELinalg.sum"(%a0) : (tensor<3x3x!FHE.eint<4>>) -> !FHE.eint<4>
        //
        //     ( [1,2,3] )
        // sum ( [4,5,6] ) = 45
        //     ( [7,8,9] )
        //
        ```

        ```mlir
        // Returns the sum of all elements of `%a0` along columns
        "FHELinalg.sum"(%a0) { axes = [0] } : (tensor<3x2x!FHE.eint<4>>) -> tensor<2x!FHE.eint<4>>
        //
        //     ( [1,2] )
        // sum ( [3,4] ) = [9, 12]
        //     ( [5,6] )
        //
        ```

        ```mlir
        // Returns the sum of all elements of `%a0` along columns while preserving dimensions
        "FHELinalg.sum"(%a0) { axes = [0], keep_dims = true } : (tensor<3x2x!FHE.eint<4>>) -> tensor<1x2x!FHE.eint<4>>
        //
        //     ( [1,2] )
        // sum ( [3,4] ) = [[9, 12]]
        //     ( [5,6] )
        //
        ```

        ```mlir
        // Returns the sum of all elements of `%a0` along rows
        "FHELinalg.sum"(%a0) { axes = [1] } : (tensor<3x2x!FHE.eint<4>>) -> tensor<3x!FHE.eint<4>>
        //
        //     ( [1,2] )
        // sum ( [3,4] ) = [3, 7, 11]
        //     ( [5,6] )
        //
        ```

        ```mlir
        // Returns the sum of all elements of `%a0` along rows while preserving dimensions
        "FHELinalg.sum"(%a0) { axes = [1], keep_dims = true } : (tensor<3x2x!FHE.eint<4>>) -> tensor<3x1x!FHE.eint<4>>
        //
        //     ( [1,2] )   [3]
        // sum ( [3,4] ) = [7]
        //     ( [5,6] )   [11]
        //
        ```
    }];

    let arguments = (ins
        Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>:$tensor,
        DefaultValuedAttr<I64ArrayAttr, "{}">:$axes,
        DefaultValuedAttr<BoolAttr, "false">:$keep_dims
    );

    let results = (outs
        TypeConstraint<Or<[
            FHE_AnyEncryptedInteger.predicate,
            And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>
        ]>>:$out
    );

    let hasVerifier = 1;
}

def FHELinalg_ConcatOp : FHELinalg_Op<"concat", [Pure]> {
    let summary = "Concatenates a sequence of tensors along an existing axis.";

    let description = [{
        Concatenates several tensors along a given axis.

        Examples:

        ```mlir
        "FHELinalg.concat"(%a, %b) { axis = 0 } : (tensor<3x3x!FHE.eint<4>>, tensor<3x3x!FHE.eint<4>>) -> tensor<6x3x!FHE.eint<4>>
        //
        //        ( [1,2,3]  [1,2,3] )   [1,2,3]
        // concat ( [4,5,6], [4,5,6] ) = [4,5,6]
        //        ( [7,8,9]  [7,8,9] )   [7,8,9]
        //                               [1,2,3]
        //                               [4,5,6]
        //                               [7,8,9]
        //
        ```

        ```mlir
        "FHELinalg.concat"(%a, %b) { axis = 1 } : (tensor<3x3x!FHE.eint<4>>, tensor<3x3x!FHE.eint<4>>) -> tensor<3x6x!FHE.eint<4>>
        //
        //        ( [1,2,3]  [1,2,3] )   [1,2,3,1,2,3]
        // concat ( [4,5,6], [4,5,6] ) = [4,5,6,4,5,6]
        //        ( [7,8,9]  [7,8,9] )   [7,8,9,7,8,9]
        //
        ```
    }];

    let arguments = (ins
        Variadic<Type<And<[TensorOf<[AnyType]>.predicate, HasStaticShapePred]>>>:$ins,
        DefaultValuedAttr<I64Attr, "0">:$axis
    );

    let results = (outs
        Type<And<[TensorOf<[AnyType]>.predicate, HasStaticShapePred]>>:$out
    );

    let hasVerifier = 1;
    let hasFolder = 1;
}

def FHELinalg_Conv2dOp : FHELinalg_Op<"conv2d", [Pure, BinaryEintInt, DeclareOpInterfaceMethods<Binary, ["sqMANP"]>]> {
  let summary = "Returns the 2D convolution of a tensor in the form NCHW with weights in the form FCHW";

  let arguments = (ins
    Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>:$input,
    Type<And<[TensorOf<[AnyInteger]>.predicate, HasStaticShapePred]>>:$weight,
    Optional<Type<And<[TensorOf<[AnyInteger]>.predicate, HasStaticShapePred]>>>:$bias,
    // Since there is no U64ElementsAttr, we use I64 and make sure there is no neg values during verification
    OptionalAttr<I64ElementsAttr>:$padding,
    OptionalAttr<I64ElementsAttr>:$strides,
    OptionalAttr<I64ElementsAttr>:$dilations,
    OptionalAttr<I64Attr>:$group
  );
  let results = (outs Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>);
  let hasVerifier = 1;
}

def FHELinalg_Maxpool2dOp : FHELinalg_Op<"maxpool2d", [UnaryEint, DeclareOpInterfaceMethods<UnaryEint, ["sqMANP"]>]> {
  let summary = "Returns the 2D maxpool of a tensor in the form NCHW";

  let arguments = (ins
    Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>:$input,
    I64ElementsAttr:$kernel_shape,
    OptionalAttr<I64ElementsAttr>:$strides,
    OptionalAttr<I64ElementsAttr>:$dilations
  );
  let results = (outs Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>);
  let hasVerifier = 1;
}

def FHELinalg_TransposeOp : FHELinalg_Op<"transpose", [Pure, UnaryEint, DeclareOpInterfaceMethods<UnaryEint>]> {
    let summary = "Returns a tensor that contains the transposition of the input tensor.";

    let description = [{
        Performs a transpose operation on an N-dimensional tensor.

        Attributes:

        - axes: I64ArrayAttr = []
            list of dimension to perform the transposition
            contains a permutation of [0,1,..,N-1] where N is the number of axes
            think of it as a way to rearrange axes (see the example below)

        ```mlir
        "FHELinalg.transpose"(%a) : (tensor<n0xn1x...xnNxType>) -> tensor<nNx...xn1xn0xType>
        ```

        Examples:
        ```mlir
        // Transpose the input tensor
        // [1,2]    [1, 3, 5]
        // [3,4] => [2, 4, 6]
        // [5,6]
        //
        "FHELinalg.transpose"(%a) : (tensor<3x2xi7>) -> tensor<2x3xi7>
        ```

        ```mlir
        "FHELinalg.transpose"(%a) { axes = [1, 3, 0, 2] } : (tensor<2x3x4x5xi7>) -> tensor<3x5x2x4xi7>
        ```
    }];

    // Args and result are both supposed to be of tensor of encrypted integers, and the verifier does check that
    let arguments = (ins AnyType:$tensor, DefaultValuedAttr<I64ArrayAttr, "{}">:$axes);
    let results = (outs AnyType);

    let hasVerifier = 1;
}

def FHELinalg_FromElementOp : FHELinalg_Op<"from_element", [Pure]> {
    let summary = "Creates a tensor with a single element.";

    let description = [{
        Creates a tensor with a single element.

        ```mlir
        "FHELinalg.from_element"(%a) : (Type) -> tensor<1xType>
        ```
    }];

    let arguments = (ins AnyType);
    let results = (outs Type<And<[TensorOf<[AnyType]>.predicate, HasStaticShapePred]>>);

    let hasVerifier = 1;
}

def FHELinalg_ToSignedOp : FHELinalg_Op<"to_signed", [Pure, UnaryEint, DeclareOpInterfaceMethods<UnaryEint>]> {
    let summary = "Cast an unsigned integer tensor to a signed one";

    let description = [{
        Cast an unsigned integer tensor to a signed one.
        The result must have the same width and the same shape as the input.

        The behavior is undefined on overflow/underflow.

        Examples:
        ```mlir
        // ok
        "FHELinalg.to_signed"(%x) : (tensor<3x2x!FHE.eint<2>>) -> tensor<3x2x!FHE.esint<2>>

        // error
        "FHELinalg.to_signed"(%x) : (tensor<3x2x!FHE.eint<2>>) -> tensor<3x2x!FHE.esint<3>>
        ```
    }];

    let arguments = (ins
        Type<And<[TensorOf<[FHE_EncryptedUnsignedIntegerType]>.predicate, HasStaticShapePred]>>:$input
    );

    let results = (outs
        Type<And<[TensorOf<[FHE_EncryptedSignedIntegerType]>.predicate, HasStaticShapePred]>>:$output
    );

    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

def FHELinalg_ToUnsignedOp : FHELinalg_Op<"to_unsigned", [Pure, UnaryEint, DeclareOpInterfaceMethods<UnaryEint>]> {
    let summary = "Cast a signed integer tensor to an unsigned one";

    let description = [{
        Cast a signed integer tensor to an unsigned one.
        The result must have the same width and the same shape as the input.

        The behavior is undefined on overflow/underflow.

        Examples:
        ```mlir
        // ok
        "FHELinalg.to_unsigned"(%x) : (tensor<3x2x!FHE.esint<2>>) -> tensor<3x2x!FHE.eint<2>>

        // error
        "FHELinalg.to_unsigned"(%x) : (tensor<3x2x!FHE.esint<2>>) -> tensor<3x2x!FHE.eint<3>>
        ```
    }];

    let arguments = (ins
        Type<And<[TensorOf<[FHE_EncryptedSignedIntegerType]>.predicate, HasStaticShapePred]>>:$input
    );

    let results = (outs
        Type<And<[TensorOf<[FHE_EncryptedUnsignedIntegerType]>.predicate, HasStaticShapePred]>>:$output
    );

    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

def FHELinalg_RoundOp : FHELinalg_Op<"round", [Pure, TensorUnaryEint, UnaryEint, DeclareOpInterfaceMethods<UnaryEint, ["sqMANP"]>]> {
    let summary = "Rounds a tensor of ciphertexts into a smaller precision.";

    let description = [{
        Assuming a ciphertext whose message is implemented over `p` bits, this
        operation rounds it to fit to `q` bits where `p>q`.

        Example:
        ```mlir
        // ok
        "FHELinalg.round"(%a): (tensor<3x!FHE.eint<6>>) -> (tensor<3x!FHE.eint<5>>)
        "FHELinalg.round"(%a): (tensor<3x!FHE.eint<5>>) -> (tensor<3x!FHE.eint<3>>)
        "FHELinalg.round"(%a): (tensor<3x!FHE.eint<3>>) -> (tensor<3x!FHE.eint<2>>)
        "FHELinalg.round"(%a): (tensor<3x!FHE.esint<3>>) -> (tensor<3x!FHE.esint<2>>)

        // error
        "FHELinalg.round"(%a): (tensor<3x!FHE.eint<6>>) -> (tensor<3x!FHE.eint<6>>)
        "FHELinalg.round"(%a): (tensor<3x!FHE.eint<4>>) -> (tensor<3x!FHE.eint<5>>)
        "FHELinalg.round"(%a): (tensor<3x!FHE.eint<4>>) -> (tensor<3x!FHE.esint<2>>)
		```
    }];

    let arguments = (ins
    	Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>:$input
    );

    let results = (outs
    	Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>:$output
    );

    let hasVerifier = 1;
    let hasFolder = 1;
}

def FHELinalg_LsbEintOp: FHELinalg_Op<"lsb", [Pure, TensorUnaryEint, UnaryEint, ConstantNoise]> {

    let summary = "Extract the lowest significant bit at a given precision.";

    let description = [{
      This operation extracts the lsb of a ciphertext tensor in a specific precision.

      Extracting only 1 bit:
      ```mlir
       // ok
       %lsb = "FHE.lsb"(%a): (tensor<1x!FHE.eint<4>)) -> (tensor<1x!FHE.eint<1>>)

      If you need to clear the lsb of the original ciphertext, you should extract to the same precision as the ciphertext.
      If you need to extract several bits, you can extract sequentially using explicit bitwidth change and bit clearing.

      Example:
      ```mlir
       // ok
       %a_lsb = "FHELinalg.lsb"(%a): (tensor<1x!FHE.eint<4>)) -> (tensor<1x!FHE.eint<4>))
       %a_lsb_cleared = "FHELinalg.sub_eint"(%a, %lsb) : (tensor<1x!FHE.eint<4>), tensor<1x!FHE.eint<4>)) -> (tensor<1x!FHE.eint<4>))
       %b = %a : tensor<1x!FHE.eint<3>>
       // now you can extract the next lsb from %b
       %b_lsb = "FHELinalg.lsb"(%b): (tensor<1x!FHE.eint<3>>) -> (tensor<1x!FHE.eint<3>>)
       // later if you need %b_lsb at the original position
       %b_lsb_as_in_a = %b_lsb : tensor<1x!FHE.eint<3>>
    ```
    }];

    let arguments = (ins Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>:$input);
    let results = (outs Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>:$output);
    let hasVerifier = 1;
}

def FHELinalg_ReinterpretPrecisionEintOp: FHELinalg_Op<"reinterpret_precision", [Pure, TensorUnaryEint, UnaryEint]> {

    let summary = "Reinterpret the ciphertext tensor with a different precision.";

    let description = [{
      It's a reinterpretation cast which changes only the precision.
      On CRT represention, it does nothing.
      On Native representation, it moves the message/noise further forward, effectively changing the precision.
      Changing to
        - a bigger precision is safe, as the crypto-parameters are chosen such that only zeros will come from the noise part.
          This is equivalent to a shift left for the value
        - a smaller precision is only safe if you clear the lowest message bits first.
          If not, you can assume small errors with high probability and frequent bigger errors, which can be contained to small errors using margins.
          This is equivalent to a shift right for the value

      Example:
      ```mlir
       // assuming a is encoded as 4bits but can be stored in 2bits
       // we can obtain a to a smaller 2 bits precision
       %shifted_a = "FHELinalg.mul_eint_intlsb"(%a, %c_4): (tensor<1x!FHE.eint<4>>) -> (tensor<1x!FHE.eint<2>>)
       %a_small_precision = "FHELinalg.reinterpret_precision"(%shifted_a, %lsb) : (tensor<1x!FHE.eint<4>>) -> (tensor<1x!FHE.eint<2>>)
    ```
    }];

    let arguments = (ins Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>:$input);
    let results = (outs Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>:$output);
    let hasVerifier = 1;
}

def FHELinalg_ChangePartitionEintOp: FHELinalg_Op<"change_partition", [Pure, TensorUnaryEint, UnaryEint]> {

    let summary = "Change partition if necessary.";

    let description = [{
      Changing the partition of a ciphertext.
      If necessary, it keyswitch the ciphertext to a different key having a different set of parameters than the original one.

      Example:
      ```mlir
        %new_eint = "FHE.change_partition"(%eint): (tensor<2x3x!FHE.eint<16>>) -> (tensor<2x3x!FHE.eint<16>>)
    }];

    let arguments = (ins
        Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>:$input,
        OptionalAttr<FHE_PartitionAttr>:$src,
        OptionalAttr<FHE_PartitionAttr>:$dest
    );
    let results = (outs Type<And<[TensorOf<[FHE_AnyEncryptedInteger]>.predicate, HasStaticShapePred]>>:$output);
    let hasVerifier = 1;
}

def FHELinalg_FancyIndexOp : FHELinalg_Op<"fancy_index", [Pure]> {
    let summary = "Index into a tensor using a tensor of indices.";

    let description = [{
        Examples:

        ```mlir
        "FHELinalg.fancy_index"(%t, %i) : (tensor<5x!FHE.eint<6>>, tensor<3xindex>) -> tensor<3x!FHE.eint<6>>
        //
        // fancy_index([10, 20, 30, 40, 50], [3, 1, 2]) = [40, 20, 30]
        //
        ```

        ```mlir
        "FHELinalg.fancy_index"(%t, %i) : (tensor<5x!FHE.eint<6>>, tensor<3x2xindex>) -> tensor<3x2x!FHE.eint<6>>
        //
        // fancy_index([10, 20, 30, 40, 50], [[3, 1], [2, 2], [0, 4]]) = [[40, 20], [30, 30], [10, 50]]
        //
        ```

        ```mlir
        "FHELinalg.fancy_index"(%t, %i) : (tensor<2x3x!FHE.eint<6>>, tensor<3x2xindex>) -> tensor<3x!FHE.eint<6>>
        //
        // fancy_index([[11, 12, 13], [21, 22, 23]], [[1, 0], [0, 2], [0, 0]]) = [21, 13, 11]
        //
        ```

        ```mlir
        "FHELinalg.fancy_index"(%t, %i) : (tensor<3x3x!FHE.eint<6>>, tensor<2x3x2xindex>) -> tensor<2x3x!FHE.eint<6>>
        //
        // fancy_index([[11, 12, 13], [21, 22, 23], [31, 32, 33]], [[[1, 0], [0, 2], [0, 0]], [[2, 0], [1, 1], [2, 1]]]) = [[21, 13, 11], [31, 22, 32]]
        //
        ```
    }];

    let arguments = (ins
        Type<And<[TensorOf<[AnyType]>.predicate, HasStaticShapePred]>>:$input,
        Type<And<[TensorOf<[Index]>.predicate, HasStaticShapePred]>>:$indices
    );

    let results = (outs
        Type<And<[TensorOf<[AnyType]>.predicate, HasStaticShapePred]>>:$output
    );

    let hasVerifier = 1;
}

def FHELinalg_FancyAssignOp : FHELinalg_Op<"fancy_assign", [Pure]> {
    let summary = "Assigns a tensor into another tensor at a tensor of indices.";

    let description = [{
        Examples:

        ```mlir
        "FHELinalg.fancy_assign"(%t, %i, %a) : (tensor<5x!FHE.eint<16>>, tensor<3xindex>, tensor<3x!FHE.eint<16>>) -> tensor<5x!FHE.eint<16>>
        //
        // fancy_assign([10, 20, 30, 40, 50], [3, 1, 2], [1000, 2000, 3000]) = [10, 2000, 3000, 1000, 50]
        //
        ```

        ```mlir
        "FHELinalg.fancy_assign"(%t, %i, %a) : (tensor<5x!FHE.eint<16>>, tensor<2x2xindex>, tensor<2x2x!FHE.eint<16>>) -> tensor<5x!FHE.eint<16>>
        //
        // fancy_assign([10, 20, 30, 40, 50], [[3, 1], [2, 0]], [[1000, 2000], [3000, 4000]]) = [4000, 2000, 3000, 1000, 50]
        //
        ```

        ```mlir
        "FHELinalg.fancy_assign"(%t, %i, %a) : (tensor<2x3x!FHE.eint<16>>, tensor<3x2xindex>, tensor<3x!FHE.eint<16>>) -> tensor<2x3x!FHE.eint<16>>
        //
        // fancy_assign([[11, 12, 13], [21, 22, 23]], [[1, 0], [0, 2], [0, 0]], [1000, 2000, 3000]) = [[3000, 2000, 13], [1000, 22, 23]]
        //
        ```

        ```mlir
        "FHELinalg.fancy_assign"(%t, %i, %a) : (tensor<3x3x!FHE.eint<16>>, tensor<2x3x2xindex>, tensor<2x3x!FHE.eint<16>>) -> tensor<3x3x!FHE.eint<16>>
        //
        // fancy_assign(
        //     [[11, 12, 13], [21, 22, 23], [31, 32, 33]],
        //     [[[1, 0], [0, 2], [0, 0]], [[2, 0], [1, 1], [2, 1]]],
        //     [[1000, 2000, 3000], [4000, 5000, 6000]]
        // ) = [[3000, 2000, 13], [1000, 5000, 23], [4000, 6000, 33]]
        //
        ```

        Notes:
        - Assigning to the same output position results in undefined behavior.
    }];

    let arguments = (ins
        Type<And<[TensorOf<[AnyType]>.predicate, HasStaticShapePred]>>:$input,
        Type<And<[TensorOf<[Index]>.predicate, HasStaticShapePred]>>:$indices,
        Type<And<[TensorOf<[AnyType]>.predicate, HasStaticShapePred]>>:$values
    );

    let results = (outs
        Type<And<[TensorOf<[AnyType]>.predicate, HasStaticShapePred]>>:$output
    );

    let hasVerifier = 1;
}

def FHELinalg_BroadcastOp: FHELinalg_Op<"broadcast", [Pure, ConstantNoise]> {
    let summary = "Broadcasts a tensor to a shape.";
    let description = [{
        Broadcasting is used for expanding certain dimensions of a tensor
        or adding new dimensions to it at the beginning.

        An example could be broadcasting
        a tensor with shape    <1x2x1x4x1>
        to a tensor of shape <6x1x2x3x4x5>.

        In this example:
        - last dimension of the input (1) is expanded to (5)
        - the dimension before that (4) is kept
        - the dimension before that (1) is expanded to (3)
        - the dimension before that (2) is kept
        - the dimension before that (1) is kept
        - a new dimension (6) is added to the beginning

        See https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules
        for the semantics of broadcasting.

        Examples:

        ```mlir
        "FHELinalg.broadcast"(%t) : (tensor<1xindex>) -> tensor<3xindex>
        //
        // broadcast([5]) = [5, 5, 5]
        //
        ```

        ```mlir
        "FHELinalg.broadcast"(%t) : (tensor<1xindex>) -> tensor<3x2xindex>
        //
        // broadcast([5]) = [[5, 5], [5, 5], [5, 5]]
        //
        ```

        ```mlir
        "FHELinalg.broadcast"(%t) : (tensor<2xindex>) -> tensor<3x2xindex>
        //
        // broadcast([2, 6]) = [[2, 6], [2, 6], [2, 6]]
        //
        ```

        ```mlir
        "FHELinalg.broadcast"(%t) : (tensor<3x1xindex>) -> tensor<3x2xindex>
        //
        // broadcast([[1], [2], [3]]) = [[1, 1], [2, 2], [3, 3]]
        //
        ```

        ```mlir
        "FHELinalg.broadcast"(%t) : (tensor<2xindex>) -> tensor<2x3x2xindex>
        //
        // broadcast([2, 6]) = [[[2, 6], [2, 6], [2, 6]], [[2, 6], [2, 6], [2, 6]]]
        //
        ```

        ```mlir
        "FHELinalg.broadcast"(%t) : (tensor<3x1xindex>) -> tensor<2x3x2xindex>
        //
        // broadcast([[1], [2], [3]]) = [[[1, 1], [2, 2], [3, 3]], [[1, 1], [2, 2], [3, 3]]]
        //
        ```
    }];

    let arguments = (ins Type<And<[TensorOf<[AnyType]>.predicate, HasStaticShapePred]>>:$input);
    let results = (outs Type<And<[TensorOf<[AnyType]>.predicate, HasStaticShapePred]>>:$output);

    let hasVerifier = 1;
}


#endif
