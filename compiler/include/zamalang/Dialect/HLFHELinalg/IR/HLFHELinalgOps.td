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
        "HLFHELinalg.add_eint_int(%a0, %a1)" : (tensor<3x4x!HLFHE.eint<4>>, tensor<3x1xi5>) -> tensor<3x3x!HLFHE.eint<4>>

        // Returns the addition of a 3x3 matrix of encrypted integers and a 1x3 matrix (a line) of integers.
        //
        // [1,2,3]             [2,4,6]
        // [4,5,6] + [1,2,3] = [5,7,9]
        // [7,8,9]             [8,10,12]
        //
        // The dimension #2 of operand #2 is stretched as it is equals to 1.
        "HLFHELinalg.add_eint_int(%a0, %a1)" : (tensor<3x4x!HLFHE.eint<4>>, tensor<1x3xi5>) -> tensor<3x3x!HLFHE.eint<4>>

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

#endif
