#ifndef ZAMALANG_DIALECT_HLFHELinalg_IR_HLFHELinalg_OPS
#define ZAMALANG_DIALECT_HLFHELinalg_IR_HLFHELinalg_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

include "zamalang/Dialect/HLFHELinalg/IR/HLFHELinalgDialect.td"
include "zamalang/Dialect/HLFHELinalg/IR/HLFHELinalgTypes.td"

class HLFHELinalg_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<HLFHELinalg_Dialect, mnemonic, traits>;

// TensorBroadcastingRules verify that the operands and result verify the broadcasting rules
def TensorBroadcastingRules : NativeOpTrait<"TensorBroadcastingRules">;
def TensorBinaryEintInt : NativeOpTrait<"TensorBinaryEintInt">;
def TensorBinaryIntEint : NativeOpTrait<"TensorBinaryIntEint">;
def TensorBinaryEint : NativeOpTrait<"TensorBinaryEint">;


def AddEintIntOp : HLFHELinalg_Op<"add_eint_int", [TensorBroadcastingRules, TensorBinaryEintInt]> {
    let summary = "Returns a tensor that contains the addition of a tensor of encrypted integers and a tensor of clear integers.";

    let description = [{
        Performs an addition follwing the broadcasting rules between a tensor of encrypted integers and a tensor of clear integers.
        The width of the clear integers should be less or equals than the witdh of encrypted integers.

        Examples:
        ```mlir
        // Returns the term to term addition of `%a0` with `%a1`
        "HLFHELinalg.add_eint_int"(%a0, %a1) : (tensor<4x!HLFHE.eint<4>>, tensor<4xi5>) -> tensor<4x!HLFHE.eint<4>>

        // Returns the term to term addition of `%a0` with `%a1`, where dimensions equal to one are stretched.
        "HLFHELinalg.add_eint_int"(%a0, %a1) : (tensor<4x1x4x!HLFHE.eint<4>>, tensor<1x4x4xi5>) -> tensor<4x4x4x!HLFHE.eint<4>>

        // Returns the addition of a 3x3 matrix of encrypted integers and a 3x1 matrix (a column) of integers.
        //
        // [1,2,3]   [1]   [2,3,4]
        // [4,5,6] + [2] = [6,7,8]
        // [7,8,9]   [3]   [10,11,12]
        //
        // The dimension #1 of operand #2 is stretched as it is equals to 1.
        "HLFHELinalg.add_eint_int"(%a0, %a1) : (tensor<3x3x!HLFHE.eint<4>>, tensor<3x1xi5>) -> tensor<3x3x!HLFHE.eint<4>>

        // Returns the addition of a 3x3 matrix of encrypted integers and a 1x3 matrix (a line) of integers.
        //
        // [1,2,3]             [2,4,6]
        // [4,5,6] + [1,2,3] = [5,7,9]
        // [7,8,9]             [8,10,12]
        //
        // The dimension #2 of operand #2 is stretched as it is equals to 1.
        "HLFHELinalg.add_eint_int"(%a0, %a1) : (tensor<3x3x!HLFHE.eint<4>>, tensor<1x3xi5>) -> tensor<3x3x!HLFHE.eint<4>>

        // Same behavior than the previous one, but as the dimension #2 is missing of operand #2.
        "HLFHELinalg.add_eint_int(%a0, %a1)" : (tensor<3x4x!HLFHE.eint<4>>, tensor<3xi5>) -> tensor<4x4x4x!HLFHE.eint<4>>

        ```
    }];

    let arguments = (ins
        Type<And<[TensorOf<[EncryptedIntegerType]>.predicate, HasStaticShapePred]>>:$lhs,
        Type<And<[TensorOf<[AnyInteger]>.predicate, HasStaticShapePred]>>:$rhs
    );

    let results = (outs Type<And<[TensorOf<[EncryptedIntegerType]>.predicate, HasStaticShapePred]>>);

    let builders = [
        OpBuilder<(ins "Value":$rhs, "Value":$lhs), [{
         build($_builder, $_state, rhs.getType(), rhs, lhs);
        }]>
    ];
}

def AddEintOp : HLFHELinalg_Op<"add_eint", [TensorBroadcastingRules, TensorBinaryEint]> {
    let summary = "Returns a tensor that contains the addition of two tensor of encrypted integers.";

    let description = [{
        Performs an addition follwing the broadcasting rules between two tensors of encrypted integers.
        The width of the encrypted integers should be equals.

        Examples:
        ```mlir
        // Returns the term to term addition of `%a0` with `%a1`
        "HLFHELinalg.add_eint"(%a0, %a1) : (tensor<4x!HLFHE.eint<4>>, tensor<4x!HLFHE.eint<4>>) -> tensor<4x!HLFHE.eint<4>>

        // Returns the term to term addition of `%a0` with `%a1`, where dimensions equal to one are stretched.
        "HLFHELinalg.add_eint"(%a0, %a1) : (tensor<4x1x4x!HLFHE.eint<4>>, tensor<1x4x4x!HLFHE.eint<4>>) -> tensor<4x4x4x!HLFHE.eint<4>>

        // Returns the addition of a 3x3 matrix of encrypted integers and a 3x1 matrix (a column) of encrypted integers.
        //
        // [1,2,3]   [1]   [2,3,4]
        // [4,5,6] + [2] = [6,7,8]
        // [7,8,9]   [3]   [10,11,12]
        //
        // The dimension #1 of operand #2 is stretched as it is equals to 1.
        "HLFHELinalg.add_eint"(%a0, %a1) : (tensor<3x3x!HLFHE.eint<4>>, tensor<3x1x!HLFHE.eint<4>>) -> tensor<3x3x!HLFHE.eint<4>>

        // Returns the addition of a 3x3 matrix of encrypted integers and a 1x3 matrix (a line) of encrypted integers.
        //
        // [1,2,3]             [2,4,6]
        // [4,5,6] + [1,2,3] = [5,7,9]
        // [7,8,9]             [8,10,12]
        //
        // The dimension #2 of operand #2 is stretched as it is equals to 1.
        "HLFHELinalg.add_eint"(%a0, %a1) : (tensor<3x3x!HLFHE.eint<4>>, tensor<1x3x!HLFHE.eint<4>>) -> tensor<3x3x!HLFHE.eint<4>>

        // Same behavior than the previous one, but as the dimension #2 of operand #2 is missing.
        "HLFHELinalg.add_eint"(%a0, %a1) : (tensor<3x3x!HLFHE.eint<4>>, tensor<3x!HLFHE.eint<4>>) -> tensor<3x3x!HLFHE.eint<4>>
        ```
    }];

    let arguments = (ins
        Type<And<[TensorOf<[EncryptedIntegerType]>.predicate, HasStaticShapePred]>>:$lhs,
        Type<And<[TensorOf<[EncryptedIntegerType]>.predicate, HasStaticShapePred]>>:$rhs
    );

    let results = (outs Type<And<[TensorOf<[EncryptedIntegerType]>.predicate, HasStaticShapePred]>>);

    let builders = [
        OpBuilder<(ins "Value":$rhs, "Value":$lhs), [{
         build($_builder, $_state, rhs.getType(), rhs, lhs);
        }]>
    ];
}

def SubIntEintOp : HLFHELinalg_Op<"sub_int_eint", [TensorBroadcastingRules, TensorBinaryIntEint]> {
    let summary = "Returns a tensor that contains the substraction of a tensor of clear integers and a tensor of encrypted integers.";

    let description = [{
        Performs a substraction following the broadcasting rules between a tensor of clear integers and a tensor of encrypted integers.
        The width of the clear integers should be less or equals than the witdh of encrypted integers.

        Examples:
        ```mlir
        // Returns the term to term substraction of `%a0` with `%a1`
        "HLFHELinalg.sub_int_eint"(%a0, %a1) : (tensor<4xi5>, tensor<4x!HLFHE.eint<4>>) -> tensor<4x!HLFHE.eint<4>>

        // Returns the term to term substraction of `%a0` with `%a1`, where dimensions equal to one are stretched.
        "HLFHELinalg.sub_int_eint"(%a0, %a1) : (tensor<4x1x4xi5>, tensor<1x4x4x!HLFHE.eint<4>>) -> tensor<4x4x4x!HLFHE.eint<4>>

        // Returns the substraction of a 3x3 matrix of integers and a 3x1 matrix (a column) of encrypted integers.
        //
        // [1,2,3]   [1]   [0,2,3]
        // [4,5,6] - [2] = [2,3,4]
        // [7,8,9]   [3]   [4,5,6]
        //
        // The dimension #1 of operand #2 is stretched as it is equals to 1.
        "HLFHELinalg.sub_int_eint"(%a0, %a1) : (tensor<3x3xi5>, tensor<3x1x!HLFHE.eint<4>>) -> tensor<3x3x!HLFHE.eint<4>>

        // Returns the substraction of a 3x3 matrix of integers and a 1x3 matrix (a line) of encrypted integers.
        //
        // [1,2,3]             [0,0,0]
        // [4,5,6] - [1,2,3] = [3,3,3]
        // [7,8,9]             [6,6,6]
        //
        // The dimension #2 of operand #2 is stretched as it is equals to 1.
        "HLFHELinalg.sub_int_eint"(%a0, %a1) : (tensor<3x3xi5>, tensor<1x3x!HLFHE.eint<4>>) -> tensor<3x3x!HLFHE.eint<4>>

        // Same behavior than the previous one, but as the dimension #2 is missing of operand #2.
        "HLFHELinalg.sub_int_eint"(%a0, %a1) : (tensor<3x3xi5>, tensor<3x!HLFHE.eint<4>>) -> tensor<3x3x!HLFHE.eint<4>>

        ```
    }];

    let arguments = (ins
        Type<And<[TensorOf<[AnyInteger]>.predicate, HasStaticShapePred]>>:$lhs,
        Type<And<[TensorOf<[EncryptedIntegerType]>.predicate, HasStaticShapePred]>>:$rhs
    );

    let results = (outs Type<And<[TensorOf<[EncryptedIntegerType]>.predicate, HasStaticShapePred]>>);

    let builders = [
        OpBuilder<(ins "Value":$rhs, "Value":$lhs), [{
         build($_builder, $_state, lhs.getType(), rhs, lhs);
        }]>
    ];
}

def MulEintIntOp : HLFHELinalg_Op<"mul_eint_int", [TensorBroadcastingRules, TensorBinaryEintInt]> {
    let summary = "Returns a tensor that contains the multiplication of a tensor of encrypted integers and a tensor of clear integers.";

    let description = [{
        Performs a multiplication following the broadcasting rules between a tensor of encrypted integers and a tensor of clear integers.
        The width of the clear integers should be less or equals than the witdh of encrypted integers.

        Examples:
        ```mlir
        // Returns the term to term multiplication of `%a0` with `%a1`
        "HLFHELinalg.mul_eint_int"(%a0, %a1) : (tensor<4x!HLFHE.eint<4>>, tensor<4xi5>) -> tensor<4x!HLFHE.eint<4>>

        // Returns the term to term multiplication of `%a0` with `%a1`, where dimensions equal to one are stretched.
        "HLFHELinalg.mul_eint_int"(%a0, %a1) : (tensor<4x1x4x!HLFHE.eint<4>>, tensor<1x4x4xi5>) -> tensor<4x4x4x!HLFHE.eint<4>>

        // Returns the multiplication of a 3x3 matrix of encrypted integers and a 3x1 matrix (a column) of integers.
        //
        // [1,2,3]   [1]   [1,2,3]
        // [4,5,6] * [2] = [8,10,18]
        // [7,8,9]   [3]   [21,24,27]
        //
        // The dimension #1 of operand #2 is stretched as it is equals to 1.
        "HLFHELinalg.mul_eint_int"(%a0, %a1) : (tensor<3x3x!HLFHE.eint<4>>, tensor<3x1xi5>) -> tensor<3x3x!HLFHE.eint<4>>

        // Returns the multiplication of a 3x3 matrix of encrypted integers and a 1x3 matrix (a line) of integers.
        //
        // [1,2,3]             [2,4,6]
        // [4,5,6] * [1,2,3] = [5,7,9]
        // [7,8,9]             [8,10,12]
        //
        // The dimension #2 of operand #2 is stretched as it is equals to 1.
        "HLFHELinalg.mul_eint_int"(%a0, %a1) : (tensor<3x3x!HLFHE.eint<4>>, tensor<1x3xi5>) -> tensor<3x3x!HLFHE.eint<4>>

        // Same behavior than the previous one, but as the dimension #2 is missing of operand #2.
        "HLFHELinalg.mul_eint_int"(%a0, %a1) : (tensor<3x3x!HLFHE.eint<4>>, tensor<3xi5>) -> tensor<3x3x!HLFHE.eint<4>>

        ```
    }];

    let arguments = (ins
        Type<And<[TensorOf<[EncryptedIntegerType]>.predicate, HasStaticShapePred]>>:$lhs,
        Type<And<[TensorOf<[AnyInteger]>.predicate, HasStaticShapePred]>>:$rhs
    );

    let results = (outs Type<And<[TensorOf<[EncryptedIntegerType]>.predicate, HasStaticShapePred]>>);
}

def ApplyLookupTableEintOp : HLFHELinalg_Op<"apply_lookup_table", []> {
    let summary = "Returns a tensor that contains the result of the lookup on a table.";

    let description = [{
        Performs for each encrypted indices a lookup on a table of clear integers.

        ```mlir
        // The result of this operation, is a tensor that contains the result of the lookup on a table.
        // i.e. %res[i, ..., k] = %lut[%t[i, ..., k]]
        %res = HLFHELinalg.apply_lookup_table(%t, %lut): tensor<DNx...xD1x!HLFHE.eint<$p>>, tensor<D2^$pxi64> -> tensor<DNx...xD1x!HLFHE.eint<$p>>
        ```

        The `%lut` argument should be a tensor with one dimension, where its dimension is equals to `2^p` where `p` is the width of the encrypted integers.

        Examples:
        ```mlir

        // Returns the lookup of 3x3 matrix of encrypted indices of with 2 on a table of size 4=2² of clear integers.
        //
        // [0,1,2]                 [1,3,5]
        // [3,0,1] lut [1,3,5,7] = [7,1,3]
        // [2,3,0]                 [5,7,1]
        "HLFHELinalg.apply_lookup_table"(%t, %lut) : (tensor<3x3x!HLFHE.eint<2>>, tensor<4xi64>) -> tensor<3x3x!HLFHE.eint<3>>
        ```
    }];

    let arguments = (ins
        Type<And<[TensorOf<[EncryptedIntegerType]>.predicate, HasStaticShapePred]>>:$t,
        Type<And<[TensorOf<[AnyInteger]>.predicate, HasStaticShapePred]>>:$lut
    );

    let results = (outs Type<And<[TensorOf<[EncryptedIntegerType]>.predicate, HasStaticShapePred]>>);

    let verifier = [{
        return ::mlir::zamalang::HLFHELinalg::verifyApplyLookupTable(*this);
    }];
}

#endif
