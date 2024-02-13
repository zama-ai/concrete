//===- TFHEOps.td - TFHE dialect ops ----------------*- tablegen
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

class TFHE_BatchableBinaryOp<
   string mnemonic,
   string batched_scalar_opname,
   string scalar_batched_opname,
   string all_batched_opname,
   list<Trait> traits = []>
    : Op<TFHE_Dialect, mnemonic, traits # [Pure, BatchableOpInterface] >
{
  let extraClassDeclaration = [{
    struct BatchingVariant {
       static const unsigned ALL_BATCHED = 0;
       static const unsigned BATCHED_SCALAR = 1;
       static const unsigned SCALAR_BATCHED = 2;
    };

    unsigned getNumBatchingVariants() { return 3; }

    ::llvm::MutableArrayRef<::mlir::OpOperand> getBatchableOperands(unsigned variant) {
      switch(variant) {
        case BatchingVariant::ALL_BATCHED:
          return getOperation()->getOpOperands();
        case BatchingVariant::BATCHED_SCALAR:
          return getOperation()->getOpOperands().take_front();
        case BatchingVariant::SCALAR_BATCHED:
          return getOperation()->getOpOperands().drop_front().take_front();
      }

      llvm_unreachable("Unknown batching variant");
    }

    ::mlir::Value createBatchedOperation(unsigned variant,
                                         ::mlir::ImplicitLocOpBuilder& builder,
                                         ::mlir::ValueRange batchedOperands,
                                         ::mlir::ValueRange hoistedNonBatchableOperands) {
      ::mlir::RankedTensorType resType = ::mlir::RankedTensorType::get(
        batchedOperands[0].getType().cast<::mlir::RankedTensorType>().getShape(),
        getResult().getType());

      ::llvm::SmallVector<::mlir::Value> operands;

      switch(variant) {
        case BatchingVariant::ALL_BATCHED:
          operands = batchedOperands;
          return builder.create<}] # all_batched_opname # [{>(
            mlir::TypeRange{resType},
            operands,
            getOperation()->getAttrs());
        case BatchingVariant::BATCHED_SCALAR:
          operands.push_back(batchedOperands[0]);
          operands.push_back(hoistedNonBatchableOperands[0]);
          return builder.create<}] # batched_scalar_opname # [{>(
            mlir::TypeRange{resType},
            operands,
            getOperation()->getAttrs());
        case BatchingVariant::SCALAR_BATCHED:
          operands.push_back(hoistedNonBatchableOperands[0]);
          operands.push_back(batchedOperands[0]);
          return builder.create<}] # scalar_batched_opname # [{>(
            mlir::TypeRange{resType},
            operands,
            getOperation()->getAttrs());
      }

      llvm_unreachable("Unknown batching variant");
    }
  }];
}

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

    let hasVerifier = 1;
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
  let summary = "Returns a trivial encryption of 0";

  let arguments = (ins);
  let results = (outs TFHE_GLWECipherTextType : $out);
}

def TFHE_ZeroTensorGLWEOp : TFHE_Op<"zero_tensor", [Pure]> {
  let summary = "Returns a tensor containing trivial encryptions of 0";

  let arguments = (ins);
  let results = (outs Type<And<[TensorOf<[TFHE_GLWECipherTextType]>.predicate, HasStaticShapePred]>>:$tensor);
}

def TFHE_ABatchedAddGLWEIntOp : TFHE_Op<"batched_add_glwe_int", [Pure]> {
  let summary = "Batched version of AddGLWEIntOp";

  let arguments = (ins
    1DTensorOf<[TFHE_GLWECipherTextType]> : $ciphertexts,
    1DTensorOf<[AnyInteger]> : $plaintexts
  );

  let results = (outs 1DTensorOf<[TFHE_GLWECipherTextType]> : $result);
}

def TFHE_ABatchedAddGLWEIntCstOp : TFHE_Op<"batched_add_glwe_int_cst", [Pure]> {
  let summary = "Batched version of AddGLWEIntOp";

  let arguments = (ins
    1DTensorOf<[TFHE_GLWECipherTextType]> : $ciphertexts,
    AnyInteger : $plaintext
  );

  let results = (outs 1DTensorOf<[TFHE_GLWECipherTextType]> : $result);
}

def TFHE_ABatchedAddGLWECstIntOp : TFHE_Op<"batched_add_glwe_cst_int", [Pure]> {
  let summary = "Batched version of AddGLWEIntOp";

  let arguments = (ins
    TFHE_GLWECipherTextType : $ciphertext,
    1DTensorOf<[AnyInteger]> : $plaintexts
  );

  let results = (outs 1DTensorOf<[TFHE_GLWECipherTextType]> : $result);
}

def TFHE_AddGLWEIntOp : TFHE_BatchableBinaryOp<
  "add_glwe_int",
  "ABatchedAddGLWEIntCstOp",
  "ABatchedAddGLWECstIntOp",
  "ABatchedAddGLWEIntOp">
{
  let summary = "Returns the sum of a clear integer and an lwe ciphertext";

  let arguments = (ins TFHE_GLWECipherTextType : $a, AnyInteger : $b);
  let results = (outs TFHE_GLWECipherTextType);

  let hasVerifier = 1;
}

def TFHE_ABatchedAddGLWEOp : TFHE_Op<"batched_add_glwe", [Pure]> {
  let summary = "Batched version of AddGLWEOp";

  let arguments = (ins
    1DTensorOf<[TFHE_GLWECipherTextType]> : $ciphertexts_a,
    1DTensorOf<[TFHE_GLWECipherTextType]> : $ciphertexts_b
  );

  let results = (outs 1DTensorOf<[TFHE_GLWECipherTextType]> : $result);
}

def TFHE_AddGLWEOp : TFHE_Op<"add_glwe", [Pure, BatchableOpInterface]> {
  let summary = "Returns the sum of two lwe ciphertexts";

  let arguments = (ins TFHE_GLWECipherTextType : $a, TFHE_GLWECipherTextType : $b);
  let results = (outs TFHE_GLWECipherTextType);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::llvm::MutableArrayRef<::mlir::OpOperand> getBatchableOperands(unsigned variant) {
      return getOperation()->getOpOperands().take_front(2);
    }

    ::mlir::Value createBatchedOperation(unsigned variant,
                                         ::mlir::ImplicitLocOpBuilder& builder,
                                         ::mlir::ValueRange batchedOperands,
                                         ::mlir::ValueRange hoistedNonBatchableOperands) {
      ::mlir::RankedTensorType resType = ::mlir::RankedTensorType::get(
        batchedOperands[0].getType().cast<::mlir::RankedTensorType>().getShape(),
        getResult().getType());

      return builder.create<ABatchedAddGLWEOp>(
        mlir::TypeRange{resType},
        batchedOperands,
        getOperation()->getAttrs());
    }
  }];
}

def TFHE_SubGLWEIntOp : TFHE_Op<"sub_int_glwe", [Pure]> {
  let summary = "Substracts an integer and a GLWE ciphertext";

  let arguments = (ins AnyInteger : $a, TFHE_GLWECipherTextType : $b);
  let results = (outs TFHE_GLWECipherTextType);

  let hasVerifier = 1;
}

def TFHE_BatchedNegGLWEOp : TFHE_Op<"batched_neg_glwe", [Pure]> {
  let summary = "Batched version of NegGLWEOp";

  let arguments = (ins
    1DTensorOf<[TFHE_GLWECipherTextType]> : $ciphertexts
  );

  let results = (outs 1DTensorOf<[TFHE_GLWECipherTextType]> : $result);
}

def TFHE_NegGLWEOp : TFHE_Op<"neg_glwe", [Pure, BatchableOpInterface]> {
  let summary = "Negates a glwe ciphertext";

  let arguments = (ins TFHE_GLWECipherTextType : $a);
  let results = (outs TFHE_GLWECipherTextType);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::llvm::MutableArrayRef<::mlir::OpOperand> getBatchableOperands(unsigned variant) {
      return getOperation()->getOpOperands().take_front();
    }

    ::mlir::Value createBatchedOperation(unsigned variant,
                                         ::mlir::ImplicitLocOpBuilder& builder,
                                         ::mlir::ValueRange batchedOperands,
                                         ::mlir::ValueRange hoistedNonBatchableOperands) {
      ::mlir::RankedTensorType resType = ::mlir::RankedTensorType::get(
        batchedOperands[0].getType().cast<::mlir::RankedTensorType>().getShape(),
        getResult().getType());

      return builder.create<BatchedNegGLWEOp>(
        mlir::TypeRange{resType},
        mlir::ValueRange{batchedOperands},
        getOperation()->getAttrs());
    }
  }];
}

def TFHE_BatchedMulGLWEIntOp : TFHE_Op<"batched_mul_glwe_int", [Pure]> {
  let summary = "Batched version of MulGLWEIntOp";

  let arguments = (ins
    1DTensorOf<[TFHE_GLWECipherTextType]> : $ciphertexts,
    1DTensorOf<[AnyInteger]> : $cleartexts
  );

  let results = (outs 1DTensorOf<[TFHE_GLWECipherTextType]> : $result);
}

def TFHE_BatchedMulGLWECstIntOp : TFHE_Op<"batched_mul_glwe_cst_int", [Pure]> {
  let summary = "Batched version of MulGLWECstIntOp";

  let arguments = (ins
    TFHE_GLWECipherTextType : $ciphertext,
    1DTensorOf<[AnyInteger]>: $cleartexts
  );

  let results = (outs 1DTensorOf<[TFHE_GLWECipherTextType]> : $result);
}

def TFHE_BatchedMulGLWEIntCstOp : TFHE_Op<"batched_mul_glwe_int_cst", [Pure]> {
  let summary = "Batched version of MulGLWEIntCstOp";

  let arguments = (ins
    1DTensorOf<[TFHE_GLWECipherTextType]> : $ciphertexts,
    AnyInteger: $cleartext
  );

  let results = (outs 1DTensorOf<[TFHE_GLWECipherTextType]> : $result);
}

def TFHE_MulGLWEIntOp : TFHE_BatchableBinaryOp<
  "mul_glwe_int",
  "BatchedMulGLWEIntCstOp",
  "BatchedMulGLWECstIntOp",
  "BatchedMulGLWEIntOp">
{
  let summary = "Returns the product of a clear integer and an lwe ciphertext";

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
    ::llvm::MutableArrayRef<::mlir::OpOperand> getBatchableOperands(unsigned variant) {
      return getOperation()->getOpOperands().take_front();
    }

    ::mlir::Value createBatchedOperation(unsigned variant,
                                         ::mlir::ImplicitLocOpBuilder& builder,
                                         ::mlir::ValueRange batchedOperands,
                                         ::mlir::ValueRange hoistedNonBatchableOperands) {
      assert(batchedOperands.size() == 1);
      ::mlir::RankedTensorType resType = ::mlir::RankedTensorType::get(
        batchedOperands[0].getType().cast<::mlir::RankedTensorType>().getShape(),
        getResult().getType());

      return builder.create<BatchedKeySwitchGLWEOp>(
        mlir::TypeRange{resType},
        batchedOperands,
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

  let hasVerifier = 1;
}

def TFHE_BatchedMappedBootstrapGLWEOp : TFHE_Op<"batched_mapped_bootstrap_glwe", [Pure]> {
  let summary = "Batched version of KeySwitchGLWEOp which also batches the lookup table";

  let arguments = (ins
    1DTensorOf<[TFHE_GLWECipherTextType]> : $ciphertexts,
    2DTensorOf<[I64]> : $lookup_table,
    TFHE_BootstrapKeyAttr: $key
  );

  let results = (outs 1DTensorOf<[TFHE_GLWECipherTextType]> : $result);

  let hasVerifier = 1;
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

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    struct BatchingVariant {
       static const unsigned CIPHERTEXT_BATCHING = 0;
       static const unsigned MAPPED_LOOKUP = 1;
    };

    unsigned getNumBatchingVariants() { return 2; }

    ::llvm::MutableArrayRef<::mlir::OpOperand> getBatchableOperands(unsigned variant) {
      if(variant == BatchingVariant::CIPHERTEXT_BATCHING) {
        return getOperation()->getOpOperands().take_front(1);
      } else if(variant == BatchingVariant::MAPPED_LOOKUP) {
        return getOperation()->getOpOperands().take_front(2);
      } else {
        llvm_unreachable("Invalid batching variant");
      }
    }

    ::mlir::Value createBatchedOperation(unsigned variant,
                                         ::mlir::ImplicitLocOpBuilder& builder,
                                         ::mlir::ValueRange batchedOperands,
                                         ::mlir::ValueRange hoistedNonBatchableOperands) {
      ::mlir::RankedTensorType resType = ::mlir::RankedTensorType::get(
        batchedOperands[0].getType().cast<::mlir::RankedTensorType>().getShape(),
        getResult().getType());

      ::llvm::SmallVector<::mlir::Value> operands;
      operands.push_back(batchedOperands[0]);

      if(variant == BatchingVariant::MAPPED_LOOKUP) {
        operands.push_back(batchedOperands[1]);
      }

      operands.append(hoistedNonBatchableOperands.begin(),
                      hoistedNonBatchableOperands.end());

      if(variant == BatchingVariant::CIPHERTEXT_BATCHING) {
        return builder.create<BatchedBootstrapGLWEOp>(
          mlir::TypeRange{resType},
          operands,
          getOperation()->getAttrs());
      } if(variant == BatchingVariant::MAPPED_LOOKUP) {
        return builder.create<BatchedMappedBootstrapGLWEOp>(
          mlir::TypeRange{resType},
          operands,
          getOperation()->getAttrs());
      } else {
        llvm_unreachable("Unknown batching variant");
      }
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
