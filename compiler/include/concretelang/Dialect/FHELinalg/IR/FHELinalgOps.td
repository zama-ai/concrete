#ifndef CONCRETELANG_DIALECT_FHELinalg_IR_FHELinalg_OPS
#define CONCRETELANG_DIALECT_FHELinalg_IR_FHELinalg_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Dialect/StandardOps/IR/StandardOpsBase.td"
include "mlir/Dialect/Linalg/IR/LinalgBase.td"
include "mlir/Dialect/Linalg/IR/LinalgInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

include "concretelang/Dialect/FHELinalg/IR/FHELinalgDialect.td"
include "concretelang/Dialect/FHELinalg/IR/FHELinalgTypes.td"

class FHELinalg_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<FHELinalg_Dialect, mnemonic, traits>;

// TensorBroadcastingRules verify that the operands and result verify the broadcasting rules
def TensorBroadcastingRules : NativeOpTrait<"TensorBroadcastingRules">;
def TensorBinaryEintInt : NativeOpTrait<"TensorBinaryEintInt">;
def TensorBinaryIntEint : NativeOpTrait<"TensorBinaryIntEint">;
def TensorBinaryEint : NativeOpTrait<"TensorBinaryEint">;
def TensorUnaryEint : NativeOpTrait<"TensorUnaryEint">;


def AddEintIntOp : FHELinalg_Op<"add_eint_int", [TensorBroadcastingRules, TensorBinaryEintInt]> {
    let summary = "Returns a tensor that contains the addition of a tensor of encrypted integers and a tensor of clear integers.";

    let description = [{
        Performs an addition follwing the broadcasting rules between a tensor of encrypted integers and a tensor of clear integers.
        The width of the clear integers must be less than or equals to the witdh of encrypted integers.

        Examples:
        ```mlir
        // Returns the term to term addition of `%a0` with `%a1`
        "FHELinalg.add_eint_int"(%a0, %a1) : (tensor<4x!FHE.eint<4>>, tensor<4xi5>) -> tensor<4x!FHE.eint<4>>

        // Returns the term to term addition of `%a0` with `%a1`, where dimensions equal to one are stretched.
        "FHELinalg.add_eint_int"(%a0, %a1) : (tensor<4x1x4x!FHE.eint<4>>, tensor<1x4x4xi5>) -> tensor<4x4x4x!FHE.eint<4>>

        // Returns the addition of a 3x3 matrix of encrypted integers and a 3x1 matrix (a column) of integers.
        //
        // [1,2,3]   [1]   [2,3,4]
        // [4,5,6] + [2] = [6,7,8]
        // [7,8,9]   [3]   [10,11,12]
        //
        // The dimension #1 of operand #2 is stretched as it is equals to 1.
        "FHELinalg.add_eint_int"(%a0, %a1) : (tensor<3x3x!FHE.eint<4>>, tensor<3x1xi5>) -> tensor<3x3x!FHE.eint<4>>

        // Returns the addition of a 3x3 matrix of encrypted integers and a 1x3 matrix (a line) of integers.
        //
        // [1,2,3]             [2,4,6]
        // [4,5,6] + [1,2,3] = [5,7,9]
        // [7,8,9]             [8,10,12]
        //
        // The dimension #2 of operand #2 is stretched as it is equals to 1.
        "FHELinalg.add_eint_int"(%a0, %a1) : (tensor<3x3x!FHE.eint<4>>, tensor<1x3xi5>) -> tensor<3x3x!FHE.eint<4>>

        // Same behavior than the previous one, but as the dimension #2 is missing of operand #2.
        "FHELinalg.add_eint_int(%a0, %a1)" : (tensor<3x4x!FHE.eint<4>>, tensor<3xi5>) -> tensor<4x4x4x!FHE.eint<4>>

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

def AddEintOp : FHELinalg_Op<"add_eint", [TensorBroadcastingRules, TensorBinaryEint]> {
    let summary = "Returns a tensor that contains the addition of two tensor of encrypted integers.";

    let description = [{
        Performs an addition follwing the broadcasting rules between two tensors of encrypted integers.
        The width of the encrypted integers must be equals.

        Examples:
        ```mlir
        // Returns the term to term addition of `%a0` with `%a1`
        "FHELinalg.add_eint"(%a0, %a1) : (tensor<4x!FHE.eint<4>>, tensor<4x!FHE.eint<4>>) -> tensor<4x!FHE.eint<4>>

        // Returns the term to term addition of `%a0` with `%a1`, where dimensions equal to one are stretched.
        "FHELinalg.add_eint"(%a0, %a1) : (tensor<4x1x4x!FHE.eint<4>>, tensor<1x4x4x!FHE.eint<4>>) -> tensor<4x4x4x!FHE.eint<4>>

        // Returns the addition of a 3x3 matrix of encrypted integers and a 3x1 matrix (a column) of encrypted integers.
        //
        // [1,2,3]   [1]   [2,3,4]
        // [4,5,6] + [2] = [6,7,8]
        // [7,8,9]   [3]   [10,11,12]
        //
        // The dimension #1 of operand #2 is stretched as it is equals to 1.
        "FHELinalg.add_eint"(%a0, %a1) : (tensor<3x3x!FHE.eint<4>>, tensor<3x1x!FHE.eint<4>>) -> tensor<3x3x!FHE.eint<4>>

        // Returns the addition of a 3x3 matrix of encrypted integers and a 1x3 matrix (a line) of encrypted integers.
        //
        // [1,2,3]             [2,4,6]
        // [4,5,6] + [1,2,3] = [5,7,9]
        // [7,8,9]             [8,10,12]
        //
        // The dimension #2 of operand #2 is stretched as it is equals to 1.
        "FHELinalg.add_eint"(%a0, %a1) : (tensor<3x3x!FHE.eint<4>>, tensor<1x3x!FHE.eint<4>>) -> tensor<3x3x!FHE.eint<4>>

        // Same behavior than the previous one, but as the dimension #2 of operand #2 is missing.
        "FHELinalg.add_eint"(%a0, %a1) : (tensor<3x3x!FHE.eint<4>>, tensor<3x!FHE.eint<4>>) -> tensor<3x3x!FHE.eint<4>>
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

def SubIntEintOp : FHELinalg_Op<"sub_int_eint", [TensorBroadcastingRules, TensorBinaryIntEint]> {
    let summary = "Returns a tensor that contains the substraction of a tensor of clear integers and a tensor of encrypted integers.";

    let description = [{
        Performs a substraction following the broadcasting rules between a tensor of clear integers and a tensor of encrypted integers.
        The width of the clear integers must be less than or equals to the witdh of encrypted integers.

        Examples:
        ```mlir
        // Returns the term to term substraction of `%a0` with `%a1`
        "FHELinalg.sub_int_eint"(%a0, %a1) : (tensor<4xi5>, tensor<4x!FHE.eint<4>>) -> tensor<4x!FHE.eint<4>>

        // Returns the term to term substraction of `%a0` with `%a1`, where dimensions equal to one are stretched.
        "FHELinalg.sub_int_eint"(%a0, %a1) : (tensor<4x1x4xi5>, tensor<1x4x4x!FHE.eint<4>>) -> tensor<4x4x4x!FHE.eint<4>>

        // Returns the substraction of a 3x3 matrix of integers and a 3x1 matrix (a column) of encrypted integers.
        //
        // [1,2,3]   [1]   [0,2,3]
        // [4,5,6] - [2] = [2,3,4]
        // [7,8,9]   [3]   [4,5,6]
        //
        // The dimension #1 of operand #2 is stretched as it is equals to 1.
        "FHELinalg.sub_int_eint"(%a0, %a1) : (tensor<3x3xi5>, tensor<3x1x!FHE.eint<4>>) -> tensor<3x3x!FHE.eint<4>>

        // Returns the substraction of a 3x3 matrix of integers and a 1x3 matrix (a line) of encrypted integers.
        //
        // [1,2,3]             [0,0,0]
        // [4,5,6] - [1,2,3] = [3,3,3]
        // [7,8,9]             [6,6,6]
        //
        // The dimension #2 of operand #2 is stretched as it is equals to 1.
        "FHELinalg.sub_int_eint"(%a0, %a1) : (tensor<3x3xi5>, tensor<1x3x!FHE.eint<4>>) -> tensor<3x3x!FHE.eint<4>>

        // Same behavior than the previous one, but as the dimension #2 is missing of operand #2.
        "FHELinalg.sub_int_eint"(%a0, %a1) : (tensor<3x3xi5>, tensor<3x!FHE.eint<4>>) -> tensor<3x3x!FHE.eint<4>>

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

def NegEintOp : FHELinalg_Op<"neg_eint", [TensorUnaryEint]> {
    let summary = "Returns a tensor that contains the negation of a tensor of encrypted integers.";

    let description = [{
        Performs a negation to a tensor of encrypted integers.

        Examples:
        ```mlir
        // Returns the term to term negation of `%a0`
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
        Type<And<[TensorOf<[EncryptedIntegerType]>.predicate, HasStaticShapePred]>>:$tensor
    );

    let results = (outs Type<And<[TensorOf<[EncryptedIntegerType]>.predicate, HasStaticShapePred]>>);

    let builders = [
        OpBuilder<(ins "Value":$tensor), [{
         build($_builder, $_state, tensor.getType(), tensor);
        }]>
    ];
}

def MulEintIntOp : FHELinalg_Op<"mul_eint_int", [TensorBroadcastingRules, TensorBinaryEintInt]> {
    let summary = "Returns a tensor that contains the multiplication of a tensor of encrypted integers and a tensor of clear integers.";

    let description = [{
        Performs a multiplication following the broadcasting rules between a tensor of encrypted integers and a tensor of clear integers.
        The width of the clear integers must be less than or equals to the witdh of encrypted integers.

        Examples:
        ```mlir
        // Returns the term to term multiplication of `%a0` with `%a1`
        "FHELinalg.mul_eint_int"(%a0, %a1) : (tensor<4x!FHE.eint<4>>, tensor<4xi5>) -> tensor<4x!FHE.eint<4>>

        // Returns the term to term multiplication of `%a0` with `%a1`, where dimensions equal to one are stretched.
        "FHELinalg.mul_eint_int"(%a0, %a1) : (tensor<4x1x4x!FHE.eint<4>>, tensor<1x4x4xi5>) -> tensor<4x4x4x!FHE.eint<4>>

        // Returns the multiplication of a 3x3 matrix of encrypted integers and a 3x1 matrix (a column) of integers.
        //
        // [1,2,3]   [1]   [1,2,3]
        // [4,5,6] * [2] = [8,10,18]
        // [7,8,9]   [3]   [21,24,27]
        //
        // The dimension #1 of operand #2 is stretched as it is equals to 1.
        "FHELinalg.mul_eint_int"(%a0, %a1) : (tensor<3x3x!FHE.eint<4>>, tensor<3x1xi5>) -> tensor<3x3x!FHE.eint<4>>

        // Returns the multiplication of a 3x3 matrix of encrypted integers and a 1x3 matrix (a line) of integers.
        //
        // [1,2,3]             [2,4,6]
        // [4,5,6] * [1,2,3] = [5,7,9]
        // [7,8,9]             [8,10,12]
        //
        // The dimension #2 of operand #2 is stretched as it is equals to 1.
        "FHELinalg.mul_eint_int"(%a0, %a1) : (tensor<3x3x!FHE.eint<4>>, tensor<1x3xi5>) -> tensor<3x3x!FHE.eint<4>>

        // Same behavior than the previous one, but as the dimension #2 is missing of operand #2.
        "FHELinalg.mul_eint_int"(%a0, %a1) : (tensor<3x3x!FHE.eint<4>>, tensor<3xi5>) -> tensor<3x3x!FHE.eint<4>>

        ```
    }];

    let arguments = (ins
        Type<And<[TensorOf<[EncryptedIntegerType]>.predicate, HasStaticShapePred]>>:$lhs,
        Type<And<[TensorOf<[AnyInteger]>.predicate, HasStaticShapePred]>>:$rhs
    );

    let results = (outs Type<And<[TensorOf<[EncryptedIntegerType]>.predicate, HasStaticShapePred]>>);
}

def ApplyLookupTableEintOp : FHELinalg_Op<"apply_lookup_table", []> {
    let summary = "Returns a tensor that contains the result of the lookup on a table.";

    let description = [{
        Performs for each encrypted indices a lookup on a table of clear integers.

        ```mlir
        // The result of this operation, is a tensor that contains the result of the lookup on a table.
        // i.e. %res[i, ..., k] = %lut[%t[i, ..., k]]
        %res = FHELinalg.apply_lookup_table(%t, %lut): tensor<DNx...xD1x!FHE.eint<$p>>, tensor<D2^$pxi64> -> tensor<DNx...xD1x!FHE.eint<$p>>
        ```

        The `%lut` argument must be a tensor with one dimension, where its dimension is equals to `2^p` where `p` is the width of the encrypted integers.

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
        Type<And<[TensorOf<[EncryptedIntegerType]>.predicate, HasStaticShapePred]>>:$t,
        Type<And<[TensorOf<[AnyInteger]>.predicate, HasStaticShapePred]>>:$lut
    );

    let results = (outs Type<And<[TensorOf<[EncryptedIntegerType]>.predicate, HasStaticShapePred]>>);

    let verifier = [{
        return ::mlir::concretelang::FHELinalg::verifyApplyLookupTable(*this);
    }];
}

def ApplyMultiLookupTableEintOp : FHELinalg_Op<"apply_multi_lookup_table", []> {
    let summary = "Returns a tensor that contains the result of the lookup on a table, using a different lookup table for each element.";

    let description = [{
        Performs for each encrypted indice a lookup on a table of clear integers. Multiple lookup tables are passed, and the application of lookup tables
        is performed following the broadcasting rules.

        ```mlir
        // The result of this operation, is a tensor that contains the result of the lookup on different tables.
        // i.e. %res[i, ..., k] = [ %luts[i][%t[i]], ..., %luts[k][%t[k]] ]
        %res = FHELinalg.apply_multi_lookup_table(%t, %lut): tensor<DNx...xD1x!FHE.eint<$p>>, tensor<DMx...xD1xD2^$pxi64> -> tensor<DNx...xD1x!FHE.eint<$p>>
        ```

        The `%luts` argument should be a tensor with M dimension, where the first M-1 dimensions are broadcastable with the N dimensions of the encrypted tensor,
        and where the last dimension dimension is equals to `2^p` where `p` is the width of the encrypted integers.

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
        Type<And<[TensorOf<[EncryptedIntegerType]>.predicate, HasStaticShapePred]>>:$t,
        Type<And<[TensorOf<[AnyInteger]>.predicate, HasStaticShapePred]>>:$luts
    );

    let results = (outs Type<And<[TensorOf<[EncryptedIntegerType]>.predicate, HasStaticShapePred]>>);

    let verifier = [{
        return ::mlir::concretelang::FHELinalg::verifyApplyMultiLookupTable(*this);
    }];
}

def ApplyMappedLookupTableEintOp : FHELinalg_Op<"apply_mapped_lookup_table", []> {
    let summary = "Returns a tensor that contains the result of the lookup on a table, using a different lookup table for each element, specified by a map.";

    let description = [{
        Performs for each encrypted indice a lookup on a table of clear integers. Multiple lookup tables are passed, and the application of lookup tables
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
        Type<And<[TensorOf<[EncryptedIntegerType]>.predicate, HasStaticShapePred]>>:$t,
        Type<And<[TensorOf<[AnyInteger]>.predicate, HasStaticShapePred]>>:$luts,
        Type<And<[TensorOf<[Index]>.predicate, HasStaticShapePred]>>:$map
    );

    let results = (outs Type<And<[TensorOf<[EncryptedIntegerType]>.predicate, HasStaticShapePred]>>);

    let verifier = [{
        return ::mlir::concretelang::FHELinalg::verifyApplyMappedLookupTable(*this);
    }];
}

def Dot : FHELinalg_Op<"dot_eint_int"> {
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
        Type<And<[TensorOf<[EncryptedIntegerType]>.predicate, HasStaticShapePred, HasAnyRankOfPred<[1]>]>>:$lhs,
        Type<And<[TensorOf<[AnyInteger]>.predicate, HasStaticShapePred, HasAnyRankOfPred<[1]>]>>:$rhs);

    let results = (outs EncryptedIntegerType:$out);

    let verifier = [{
        return ::mlir::concretelang::FHELinalg::verifyDotEintInt(*this);
    }];
}

def MatMulEintIntOp : FHELinalg_Op<"matmul_eint_int", [TensorBinaryEintInt]> {
    let summary = "Returns a tensor that contains the result of the matrix multiplication of a matrix of encrypted integers and a matrix of clear integers.";

    let description = [{
        Performs a matrix multiplication of a matrix of encrypted integers and a matrix of clear integers.
        The width of the clear integers must be less than or equals to the witdh of encrypted integers.

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
        Type<And<[TensorOf<[EncryptedIntegerType]>.predicate, HasStaticShapePred]>>:$lhs,
        Type<And<[TensorOf<[AnyInteger]>.predicate, HasStaticShapePred]>>:$rhs
    );

    let results = (outs Type<And<[TensorOf<[EncryptedIntegerType]>.predicate, HasStaticShapePred]>>);

    let verifier = [{
        return ::mlir::concretelang::FHELinalg::verifyMatmul<mlir::concretelang::FHELinalg::MatMulEintIntOp>(*this);
    }];
}

def MatMulIntEintOp : FHELinalg_Op<"matmul_int_eint", [TensorBinaryIntEint]> {
    let summary = "Returns a tensor that contains the result of the matrix multiplication of a matrix of clear integers and a matrix of encrypted integers.";

    let description = [{
        Performs a matrix multiplication of a matrix of clear integers and a matrix of encrypted integers.
        The width of the clear integers must be less than or equals to the witdh of encrypted integers.

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
        Type<And<[TensorOf<[EncryptedIntegerType]>.predicate, HasStaticShapePred]>>:$rhs
    );

    let results = (outs Type<And<[TensorOf<[EncryptedIntegerType]>.predicate, HasStaticShapePred]>>);

    let verifier = [{
        return ::mlir::concretelang::FHELinalg::verifyMatmul<mlir::concretelang::FHELinalg::MatMulIntEintOp>(*this);
    }];
}

def SumOp : FHELinalg_Op<"sum", [TensorUnaryEint]> {
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
        Type<And<[TensorOf<[EncryptedIntegerType]>.predicate, HasStaticShapePred]>>:$tensor,
        DefaultValuedAttr<I64ArrayAttr, "{}">:$axes,
        DefaultValuedAttr<BoolAttr, "false">:$keep_dims
    );

    let results = (outs
        TypeConstraint<Or<[
            EncryptedIntegerType.predicate,
            And<[TensorOf<[EncryptedIntegerType]>.predicate, HasStaticShapePred]>
        ]>>:$out
    );

    let verifier = [{
        return mlir::concretelang::FHELinalg::verifySum(*this);
    }];
}

def ConcatOp : FHELinalg_Op<"concat"> {
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
        Variadic<Type<And<[TensorOf<[EncryptedIntegerType]>.predicate, HasStaticShapePred]>>>:$ins,
        DefaultValuedAttr<I64Attr, "0">:$axis
    );

    let results = (outs
        Type<And<[TensorOf<[EncryptedIntegerType]>.predicate, HasStaticShapePred]>>:$out
    );

    let verifier = [{
        return mlir::concretelang::FHELinalg::verifyConcat(*this);
    }];
}

def FHELinalg_Conv2dOp : FHELinalg_Op<"conv2d", []> {
  let summary = "Returns the 2D convolution of a tensor in the form NCHW with weights in the form FCHW";
  let arguments = (ins
    Type<And<[TensorOf<[EncryptedIntegerType]>.predicate, HasStaticShapePred]>>:$input,
    Type<And<[TensorOf<[AnyInteger]>.predicate, HasStaticShapePred]>>:$weight,
    Optional<Type<And<[TensorOf<[AnyInteger]>.predicate, HasStaticShapePred]>>>:$bias,
    // Since there is no U64ElementsAttr, we use I64 and make sure there is no neg values during verification
    OptionalAttr<I64ElementsAttr>:$padding,
    OptionalAttr<I64ElementsAttr>:$strides,
    OptionalAttr<I64ElementsAttr>:$dilations
  );
  let results = (outs Type<And<[TensorOf<[EncryptedIntegerType]>.predicate, HasStaticShapePred]>>);
  let verifier = [{
        return ::mlir::concretelang::FHELinalg::verifyConv2d(*this);
    }];
}

class LinalgStructuredBase_Op<string mnemonic, list<OpTrait> props>
  : Op<Linalg_Dialect, mnemonic, !listconcat([
       SingleBlockImplicitTerminator<"YieldOp">,
       DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
       LinalgStructuredInterface,
       ReifyRankedShapedTypeOpInterface], props)> {
  code structuredOpsBaseDecls = [{
    // Return whether the op accesses the iteration indices.
    bool hasIndexSemantics() {
      return !this->getBody()->getOps<IndexOp>().empty();
    }

    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
      return cast<LinalgOp>(getOperation()).reifyResultShapes(b,
          reifiedReturnShapes);
    }
  }];
}

def FhelinalgConv2DNchwFchwOp : LinalgStructuredBase_Op<"fhelinalg_conv_2d_nchw_fchw", !listconcat([AttrSizedOperandSegments],
  /*extraInterfaces=*/[LinalgConvolutionOpInterface])> {

  let cppNamespace = "mlir::concretelang::FHELinalg";
  let summary = [{ Performs 2-D convolution. }];
  let description = [{
    Layout:
  * Input: NCHW.
  * Kernel: FCHW.

Numeric casting is performed on the operands to the inner multiply, promoting
them to the same data type as the accumulator/output.
  }];

    let arguments = (ins
      Variadic<AnyType>:$inputs,
      Variadic<AnyShaped>:$outputs,
RankedI64ElementsAttr<[2]>:$strides,
RankedI64ElementsAttr<[2]>:$dilations
    );
    let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
    let regions = (region AnyRegion:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<
      (ins "ValueRange":$inputs, "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        SmallVector<Type> resultTensorTypes;
        copy_if(outputs.getTypes(),
                std::back_inserter(resultTensorTypes),
                [](Type type) { return type.isa<RankedTensorType>(); });
        $_state.addTypes(resultTensorTypes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        $_state.addAttributes(attributes);
        createAndFillStructuredOpRegion<FhelinalgConv2DNchwFchwOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
            "ValueRange":$outputs,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(inputs);
        $_state.addOperands(outputs);
        $_state.addTypes(resultTensorTypes);
        $_state.addAttributes(attributes);
        $_state.addAttribute(
          "operand_segment_sizes",
          $_builder.getI32VectorAttr({
            static_cast<int32_t>(inputs.size()),
            static_cast<int32_t>(outputs.size())}));
        createAndFillStructuredOpRegion<FhelinalgConv2DNchwFchwOp>(
          $_builder,
          $_state,
          TypeRange(inputs),
          TypeRange(outputs));
      }]>,
      OpBuilder<
      (ins "TypeRange":$resultTensorTypes, "ValueRange":$operands,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
      [{
        $_state.addOperands(operands);
        $_state.addAttributes(attributes);
        $_state.addTypes(resultTensorTypes);
        (void)$_state.addRegion();
      }]>

  , OpBuilder<
  (ins "TypeRange":$resultTensorTypes, "ValueRange":$inputs,
       "ValueRange":$outputs, "Attribute":$strides, "Attribute":$dilations,
       CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(inputs);
    $_state.addOperands(outputs);
    $_state.addTypes(resultTensorTypes);
    $_state.addAttribute(
      "operand_segment_sizes",
      $_builder.getI32VectorAttr({
        static_cast<int32_t>(inputs.size()),
        static_cast<int32_t>(outputs.size())}));
    createAndFillStructuredOpRegion<FhelinalgConv2DNchwFchwOp>(
      $_builder,
      $_state,
      TypeRange(inputs),
      TypeRange(outputs));
    $_state.addAttribute("strides", strides);
$_state.addAttribute("dilations", dilations);
    $_state.addAttributes(attributes);
  }]>

    ];
    let printer = [{ return mlir::concretelang::FHELinalg::printNamedStructuredOp(p, *this); }];
    let parser = [{
      return mlir::concretelang::FHELinalg::parseNamedStructuredOp<FhelinalgConv2DNchwFchwOp>(parser, result);
    }];
    let hasFolder = 1;

    let extraClassDeclaration = structuredOpsBaseDecls # [{
      // Auto-generated.
      ArrayAttr iterator_types();
      ArrayAttr indexing_maps();
      static void regionBuilder(ImplicitLocOpBuilder &b, Block &block);
      static std::function<void(ImplicitLocOpBuilder &b, Block &)>
      getRegionBuilder() {
        return regionBuilder;
      }

      // Generic methods.
      static unsigned getNumRegionArgs();
      std::string getLibraryCallName();

      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();

    }];
}

def TransposeOp : FHELinalg_Op<"transpose", []> {
    let summary = "Returns a tensor that contains the transposition of the input tensor.";

    let description = [{
        Performs a transpose operation on an N-dimensional tensor.

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
    }];

    // Args and result are both supposed to be of tensor of enccrypted integers, and the verifier does check that
    let arguments = (ins AnyType:$tensor);
    let results = (outs AnyType);

    let verifier = [{
        return ::mlir::concretelang::FHELinalg::verifyTranspose(*this);
    }];
}


#endif
