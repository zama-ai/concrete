//===- TFHEOps.td - High level FHE dialect ops ----------------*- tablegen
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONCRETELANG_DIALECT_TFHE_IR_TFHE_OPS
#define CONCRETELANG_DIALECT_TFHE_IR_TFHE_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "concretelang/Dialect/TFHE/IR/TFHEAttrs.td"
include "concretelang/Dialect/TFHE/IR/TFHEDialect.td"
include "concretelang/Dialect/TFHE/IR/TFHETypes.td"
include "concretelang/Interfaces/BatchableInterface.td"

class TFHE_Op<string mnemonic, list<Trait> traits = []>
    : Op<TFHE_Dialect, mnemonic, traits>;

def TFHE_EncodeExpandLutForBootstrapOp : TFHE_Op<"encode_expand_lut_for_bootstrap", [Pure]> {
    let summary =
        "Encode and expand a lookup table so that it can be used for a bootstrap.";

    let arguments = (ins
        1DTensorOf<[I64]> : $input_lookup_table,
        I32Attr: $polySize,
        I32Attr: $outputBits,
        BoolAttr: $isSigned
    );

    let results = (outs 1DTensorOf<[I64]> : $result);
}

def TFHE_EncodeLutForCrtWopPBSOp : TFHE_Op<"encode_lut_for_crt_woppbs", [Pure]> {
    let summary =
        "Encode and expand a lookup table so that it can be used for a wop pbs.";

    let arguments = (ins
        1DTensorOf<[I64]> : $input_lookup_table,
        I64ArrayAttr: $crtDecomposition,
        I64ArrayAttr: $crtBits,
        I32Attr : $modulusProduct,
        BoolAttr: $isSigned
    );

    let results = (outs 2DTensorOf<[I64]> : $result);
}

def TFHE_EncodePlaintextWithCrtOp : TFHE_Op<"encode_plaintext_with_crt", [Pure]> {
    let summary =
        "Encodes a plaintext by decomposing it on a crt basis.";

    let arguments = (ins
        I64 : $input,
        I64ArrayAttr: $mods,
        I64Attr: $modsProd
    );

    let results = (outs 1DTensorOf<[I64]> : $result);
}


def TFHE_ZeroGLWEOp : TFHE_Op<"zero", [Pure]> {
  let summary = "Returns a trivial encyption of 0";

  let arguments = (ins);
  let results = (outs TFHE_GLWECipherTextType : $out);
}

def TFHE_ZeroTensorGLWEOp : TFHE_Op<"zero_tensor", [Pure]> {
  let summary = "Returns a tensor of trivial encyption of 0";

  let arguments = (ins);
  let results = (outs Type<And<[TensorOf<[TFHE_GLWECipherTextType]>.predicate, HasStaticShapePred]>>:$tensor);
}

def TFHE_AddGLWEIntOp : TFHE_Op<"add_glwe_int", [Pure]> {
  let summary = "Returns the sum of a clear integer and a lwe ciphertext";

  let arguments = (ins TFHE_GLWECipherTextType : $a, AnyInteger : $b);
  let results = (outs TFHE_GLWECipherTextType);

  let hasVerifier = 1;
}

def TFHE_AddGLWEOp : TFHE_Op<"add_glwe", [Pure]> {
  let summary = "Returns the sum of 2 lwe ciphertexts";

  let arguments = (ins TFHE_GLWECipherTextType : $a, TFHE_GLWECipherTextType : $b);
  let results = (outs TFHE_GLWECipherTextType);

  let hasVerifier = 1;
}

def TFHE_SubGLWEIntOp : TFHE_Op<"sub_int_glwe", [Pure]> {
  let summary = "Substracts an integer and a GLWE ciphertext";

  let arguments = (ins AnyInteger : $a, TFHE_GLWECipherTextType : $b);
  let results = (outs TFHE_GLWECipherTextType);

  let hasVerifier = 1;
}

def TFHE_NegGLWEOp : TFHE_Op<"neg_glwe", [Pure]> {
  let summary = "Negates a glwe ciphertext";

  let arguments = (ins TFHE_GLWECipherTextType : $a);
  let results = (outs TFHE_GLWECipherTextType);

  let hasVerifier = 1;
}

def TFHE_MulGLWEIntOp : TFHE_Op<"mul_glwe_int", [Pure]> {
  let summary = "Returns the product of a clear integer and a lwe ciphertext";

  let arguments = (ins TFHE_GLWECipherTextType : $a, AnyInteger : $b);
  let results = (outs TFHE_GLWECipherTextType);

  let hasVerifier = 1;
}

def TFHE_BatchedKeySwitchGLWEOp : TFHE_Op<"batched_keyswitch_glwe", [Pure]> {
  let summary = "Batched version of KeySwitchGLWEOp";

  let arguments = (ins
    1DTensorOf<[TFHE_GLWECipherTextType]> : $ciphertexts,
    TFHE_KeyswitchKeyAttr : $key
  );

  let results = (outs 1DTensorOf<[TFHE_GLWECipherTextType]> : $result);
}

def TFHE_KeySwitchGLWEOp : TFHE_Op<"keyswitch_glwe", [Pure, BatchableOpInterface]> {
  let summary = "Change the encryption parameters of a glwe ciphertext by "
                "applying a keyswitch";

  let arguments = (ins
    TFHE_GLWECipherTextType : $ciphertext,
    TFHE_KeyswitchKeyAttr : $key
  );


  let results = (outs TFHE_GLWECipherTextType : $result);

  let extraClassDeclaration = [{
    ::mlir::OpOperand& getBatchableOperand() {
      return getOperation()->getOpOperand(0);
    }

    ::mlir::OperandRange getNonBatchableOperands() {
      return getOperation()->getOperands().drop_front();
    }

    ::mlir::Value createBatchedOperation(::mlir::ImplicitLocOpBuilder& builder,
                                         ::mlir::Value batchedOperands,
                                         ::mlir::ValueRange hoistedNonBatchableOperands) {
      ::mlir::RankedTensorType resType = ::mlir::RankedTensorType::get(
        batchedOperands.getType().cast<::mlir::RankedTensorType>().getShape(),
        getResult().getType());

      return builder.create<BatchedKeySwitchGLWEOp>(
        mlir::TypeRange{resType},
        mlir::ValueRange{batchedOperands},
        getOperation()->getAttrs());
    }
  }];
}

def TFHE_BatchedBootstrapGLWEOp : TFHE_Op<"batched_bootstrap_glwe", [Pure]> {
  let summary = "Batched version of KeySwitchGLWEOp";

  let arguments = (ins
    1DTensorOf<[TFHE_GLWECipherTextType]> : $ciphertexts,
    1DTensorOf<[I64]> : $lookup_table,
    TFHE_BootstrapKeyAttr: $key
  );

  let results = (outs 1DTensorOf<[TFHE_GLWECipherTextType]> : $result);
}

def TFHE_BootstrapGLWEOp : TFHE_Op<"bootstrap_glwe", [Pure, BatchableOpInterface]> {
  let summary =
      "Programmable bootstraping of a GLWE ciphertext with a lookup table";

  let arguments = (ins
    TFHE_GLWECipherTextType : $ciphertext,
    1DTensorOf<[I64]> : $lookup_table,
    TFHE_BootstrapKeyAttr: $key
  );

  let results = (outs TFHE_GLWECipherTextType : $result);

  let extraClassDeclaration = [{
    ::mlir::OpOperand& getBatchableOperand() {
      return getOperation()->getOpOperand(0);
    }

    ::mlir::OperandRange getNonBatchableOperands() {
      return getOperation()->getOperands().drop_front();
    }

    ::mlir::Value createBatchedOperation(::mlir::ImplicitLocOpBuilder& builder,
                                         ::mlir::Value batchedOperands,
                                         ::mlir::ValueRange hoistedNonBatchableOperands) {
      ::mlir::RankedTensorType resType = ::mlir::RankedTensorType::get(
        batchedOperands.getType().cast<::mlir::RankedTensorType>().getShape(),
        getResult().getType());

      ::llvm::SmallVector<::mlir::Value> operands;
      operands.push_back(batchedOperands);
      operands.append(hoistedNonBatchableOperands.begin(),
                      hoistedNonBatchableOperands.end());

      return builder.create<BatchedBootstrapGLWEOp>(
        mlir::TypeRange{resType},
        operands,
        getOperation()->getAttrs());
    }
  }];
}

def TFHE_WopPBSGLWEOp : TFHE_Op<"wop_pbs_glwe", [Pure]> {
    let summary = "";

    let arguments = (ins
        Type<And<[TensorOf<[TFHE_GLWECipherTextType]>.predicate, HasStaticShapePred]>>: $ciphertexts,
        2DTensorOf<[I64]> : $lookupTable,
        TFHE_KeyswitchKeyAttr: $ksk,
        TFHE_BootstrapKeyAttr: $bsk,
        TFHE_PackingKeyswitchKeyAttr: $pksk,
        I64ArrayAttr: $crtDecomposition,
        I32Attr: $cbsLevels,
        I32Attr: $cbsBaseLog
    );

    let results = (outs Type<And<[TensorOf<[TFHE_GLWECipherTextType]>.predicate, HasStaticShapePred]>>:$result);
}


#endif
