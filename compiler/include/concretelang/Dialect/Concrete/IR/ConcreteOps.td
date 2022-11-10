#ifndef CONCRETELANG_DIALECT_Concrete_IR_Concrete_OPS
#define CONCRETELANG_DIALECT_Concrete_IR_Concrete_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

include "concretelang/Dialect/Concrete/IR/ConcreteDialect.td"
include "concretelang/Dialect/Concrete/IR/ConcreteTypes.td"
include "concretelang/Interfaces/BatchableInterface.td"

class Concrete_Op<string mnemonic, list<Trait> traits = []> :
    Op<Concrete_Dialect, mnemonic, traits>;

def Concrete_ZeroLWEOp : Concrete_Op<"zero"> {
    let summary = "Returns a trivial encyption of 0";

    let arguments = (ins);
    let results = (outs Concrete_LweCiphertextType:$out);
}

def Concrete_ZeroTensorLWEOp : Concrete_Op<"zero_tensor"> {
    let summary = "Returns a trivial encyption of 0";

    let arguments = (ins);
    let results = (outs Type<And<[TensorOf<[Concrete_LweCiphertextType]>.predicate, HasStaticShapePred]>>:$tensor);
}

def Concrete_AddLweCiphertextsOp : Concrete_Op<"add_lwe_ciphertexts"> {
    let summary = "Returns the sum of 2 lwe ciphertexts";

    let arguments = (ins Concrete_LweCiphertextType:$lhs, Concrete_LweCiphertextType:$rhs);
    let results = (outs Concrete_LweCiphertextType:$result);
}

def Concrete_AddPlaintextLweCiphertextOp : Concrete_Op<"add_plaintext_lwe_ciphertext"> {
    let summary = "Returns the sum of a clear integer and a lwe ciphertext";

    let arguments = (ins Concrete_LweCiphertextType:$lhs, AnyInteger:$rhs);
    let results = (outs Concrete_LweCiphertextType:$result);
}

def Concrete_MulCleartextLweCiphertextOp : Concrete_Op<"mul_cleartext_lwe_ciphertext"> {
    let summary = "Returns the product of a clear integer and a lwe ciphertext";

    let arguments = (ins Concrete_LweCiphertextType:$lhs, AnyInteger:$rhs);
    let results = (outs Concrete_LweCiphertextType:$result);
}

def Concrete_NegateLweCiphertextOp : Concrete_Op<"negate_lwe_ciphertext"> {
    let summary = "Negates a lwe ciphertext";

    let arguments = (ins Concrete_LweCiphertextType:$ciphertext);
    let results = (outs Concrete_LweCiphertextType:$result);
}

def Concrete_EncodeExpandLutForBootstrapOp : Concrete_Op<"encode_expand_lut_for_bootstrap"> {
    let summary =
      "Encode and expand a lookup table so that it can be used for a bootstrap.";

    let arguments = (ins
      1DTensorOf<[I64]> : $input_lookup_table,
      I32Attr: $polySize,
      I32Attr: $outputBits
    );

    let results = (outs 1DTensorOf<[I64]> : $result);
}

def Concrete_EncodeExpandLutForWopPBSOp : Concrete_Op<"encode_expand_lut_for_woppbs"> {
let summary =
    "Encode and expand a lookup table so that it can be used for a wop pbs.";

    let arguments = (ins
        1DTensorOf<[I64]> : $input_lookup_table,
        I64ArrayAttr: $crtDecomposition,
        I64ArrayAttr: $crtBits,
        I32Attr : $polySize,
        I32Attr : $modulusProduct
    );

    let results = (outs 1DTensorOf<[I64]> : $result);
}

def Concrete_EncodePlaintextWithCrtOp : Concrete_Op<"encode_plaintext_with_crt"> {
    let summary =
        "Encodes a plaintext by decomposing it on a crt basis.";

    let arguments = (ins
        I64 : $input,
        I64ArrayAttr: $mods,
        I64Attr: $modsProd
    );

    let results = (outs 1DTensorOf<[I64]> : $result);
}

def Concrete_BootstrapLweOp : Concrete_Op<"bootstrap_lwe", [BatchableOpInterface]> {
    let summary = "Bootstraps a LWE ciphertext with a GLWE trivial encryption of the lookup table";

    let arguments = (ins 
        Concrete_LweCiphertextType:$input_ciphertext,
        1DTensorOf<[I64]>:$lookup_table,
        I32Attr:$level,
        I32Attr:$baseLog,
        I32Attr:$polySize,
        I32Attr:$glweDimension
    );
    let results = (outs Concrete_LweCiphertextType:$result);

    let extraClassDeclaration = [{
      ::mlir::OpOperand& getBatchableOperand() {
        return getOperation()->getOpOperand(0);
      }

      ::mlir::OperandRange getNonBatchableOperands() {
        return getOperation()->getOperands().drop_front();
      }

      ::mlir::Value createBatchedOperation(::mlir::ImplicitLocOpBuilder& builder,
                                           ::mlir::Value batchedOperands) {
        ::mlir::RankedTensorType resType = ::mlir::RankedTensorType::get(
          batchedOperands.getType().cast<::mlir::RankedTensorType>().getShape(),
          getResult().getType());

        return builder.create<BatchedBootstrapLweOp>(
          mlir::TypeRange{resType},
          mlir::ValueRange{batchedOperands, lookup_table()},
          getOperation()->getAttrs());
      }
    }];

}

def Concrete_BatchedBootstrapLweOp : Concrete_Op<"batched_bootstrap_lwe"> {
    let summary = "Batched version of BootstrapLweOp, which performs the same operation on a tensor of elements";

    let arguments = (ins
        1DTensorOf<[Concrete_LweCiphertextType]>:$input_ciphertexts,
        1DTensorOf<[I64]>:$lookup_table,
        I32Attr:$level,
        I32Attr:$baseLog,
        I32Attr:$polySize,
        I32Attr:$glweDimension
    );
    let results = (outs 1DTensorOf<[Concrete_LweCiphertextType]>:$result);
}

def Concrete_KeySwitchLweOp : Concrete_Op<"keyswitch_lwe", [BatchableOpInterface]> {
    let summary = "Keyswitches a LWE ciphertext";

    let arguments = (ins
        Concrete_LweCiphertextType:$ciphertext,
        I32Attr:$level,
        I32Attr:$baseLog
    );
    let results = (outs Concrete_LweCiphertextType:$result);

    let extraClassDeclaration = [{
      ::mlir::OpOperand& getBatchableOperand() {
        return getOperation()->getOpOperand(0);
      }

      ::mlir::OperandRange getNonBatchableOperands() {
        return getOperation()->getOperands().drop_front();
      }

      ::mlir::Value createBatchedOperation(::mlir::ImplicitLocOpBuilder& builder,
                                           ::mlir::Value batchedOperands) {
        ::mlir::RankedTensorType resType = ::mlir::RankedTensorType::get(
          batchedOperands.getType().cast<::mlir::RankedTensorType>().getShape(),
          getResult().getType());

        return builder.create<BatchedKeySwitchLweOp>(
          mlir::TypeRange{resType},
          mlir::ValueRange{batchedOperands},
          getOperation()->getAttrs());
      }
    }];
}

def Concrete_BatchedKeySwitchLweOp : Concrete_Op<"batched_keyswitch_lwe"> {
    let summary = "Batched version of KeySwitchLweOp, which performs the same operation on a tensor of elements";

    let arguments = (ins
        1DTensorOf<[Concrete_LweCiphertextType]>:$ciphertexts,
        I32Attr:$level,
        I32Attr:$baseLog
    );
    let results = (outs 1DTensorOf<[Concrete_LweCiphertextType]>:$result);
}

// TODO(16bits): hack
def Concrete_WopPBSLweOp : Concrete_Op<"wop_pbs_lwe"> {
    let summary = "";

    let arguments = (ins
        Type<And<[TensorOf<[Concrete_LweCiphertextType]>.predicate, HasStaticShapePred]>>:$ciphertexts,
        1DTensorOf<[I64]>:$accumulator,
        // Bootstrap parameters
        I32Attr : $bootstrapLevel,
        I32Attr : $bootstrapBaseLog,
        // Keyswitch parameters
        I32Attr : $keyswitchLevel,
        I32Attr : $keyswitchBaseLog,
        // Packing keyswitch key parameters
        I32Attr : $packingKeySwitchInputLweDimension,
        I32Attr : $packingKeySwitchoutputPolynomialSize,
        I32Attr : $packingKeySwitchLevel,
        I32Attr : $packingKeySwitchBaseLog,
        // Circuit bootstrap parameters
        I32Attr : $circuitBootstrapLevel,
        I32Attr : $circuitBootstrapBaseLog,
        // Crt decomposition
        I64ArrayAttr: $crtDecomposition
    );
    let results = (outs Type<And<[TensorOf<[Concrete_LweCiphertextType]>.predicate, HasStaticShapePred]>>:$result);
}

#endif
