//===- ConcreteOps.td - Concrete dialect ops ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONCRETELANG_DIALECT_Concrete_IR_Concrete_OPS
#define CONCRETELANG_DIALECT_Concrete_IR_Concrete_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Dialect/MemRef/IR/MemRefBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

include "concretelang/Dialect/Concrete/IR/ConcreteDialect.td"
include "concretelang/Dialect/Concrete/IR/ConcreteTypes.td"
include "concretelang/Dialect/RT/IR/RTDialect.td"
include "concretelang/Dialect/RT/IR/RTTypes.td"

def Concrete_LweTensor : 1DTensorOf<[I64]>;
def Concrete_LutTensor : 1DTensorOf<[I64]>;
def Concrete_CrtLutsTensor : 2DTensorOf<[I64]>;
def Concrete_CrtPlaintextTensor : 1DTensorOf<[I64]>;
def Concrete_LweCRTTensor : 2DTensorOf<[I64]>;
def Concrete_BatchLweTensor : 2DTensorOf<[I64]>;
def Concrete_BatchPlaintextTensor : 1DTensorOf<[I64]>;
def Concrete_BatchLutTensor : 2DTensorOf<[I64]>;

def Concrete_LweBuffer : MemRefRankOf<[I64], [1]>;
def Concrete_LutBuffer : MemRefRankOf<[I64], [1]>;
def Concrete_CrtLutsBuffer : MemRefRankOf<[I64], [2]>;
def Concrete_CrtPlaintextBuffer : MemRefRankOf<[I64], [1]>;
def Concrete_LweCRTBuffer : MemRefRankOf<[I64], [2]>;
def Concrete_BatchLweBuffer : MemRefRankOf<[I64], [2]>;
def Concrete_BatchPlaintextBuffer : MemRefRankOf<[I64], [1]>;
def Concrete_BatchLutBuffer : MemRefRankOf<[I64], [2]>;

class Concrete_Op<string mnemonic, list<Trait> traits = []> :
    Op<Concrete_Dialect, mnemonic, traits>;


def Concrete_AddLweTensorOp : Concrete_Op<"add_lwe_tensor", [Pure]> {
    let summary = "Returns the sum of 2 lwe ciphertexts";

    let arguments = (ins
        Concrete_LweTensor:$lhs,
        Concrete_LweTensor:$rhs
    );
    let results = (outs Concrete_LweTensor:$result);
}

def Concrete_AddLweBufferOp : Concrete_Op<"add_lwe_buffer"> {
    let summary = "Returns the sum of 2 lwe ciphertexts";

    let arguments = (ins
        Concrete_LweBuffer:$result,
        Concrete_LweBuffer:$lhs,
        Concrete_LweBuffer:$rhs
    );
}

def Concrete_BatchedAddLweTensorOp : Concrete_Op<"batched_add_lwe_tensor", [Pure]> {
    let summary = "Batched version of AddLweTensorOp, which performs the same operation on multiple elements";

    let arguments = (ins
        Concrete_BatchLweTensor:$lhs,
        Concrete_BatchLweTensor:$rhs
    );
    let results = (outs Concrete_BatchLweTensor:$result);
}

def Concrete_BatchedAddLweBufferOp : Concrete_Op<"batched_add_lwe_buffer"> {
    let summary = "Batched version of AddLweBufferOp, which performs the same operation on multiple elements";

    let arguments = (ins
        Concrete_BatchLweBuffer:$result,
        Concrete_BatchLweBuffer:$lhs,
        Concrete_BatchLweBuffer:$rhs
    );
}

def Concrete_AddPlaintextLweTensorOp : Concrete_Op<"add_plaintext_lwe_tensor", [Pure]> {
    let summary = "Returns the sum of a clear integer and an lwe ciphertext";

    let arguments = (ins Concrete_LweTensor:$lhs, I64:$rhs);
    let results = (outs Concrete_LweTensor:$result);
}

def Concrete_AddPlaintextLweBufferOp : Concrete_Op<"add_plaintext_lwe_buffer"> {
    let summary = "Returns the sum of a clear integer and an lwe ciphertext";

    let arguments = (ins
        Concrete_LweBuffer:$result,
        Concrete_LweBuffer:$lhs,
        I64:$rhs
    );
}

def Concrete_BatchedAddPlaintextLweTensorOp : Concrete_Op<"batched_add_plaintext_lwe_tensor", [Pure]> {
    let summary = "Batched version of AddPlaintextLweTensorOp, which performs the same operation on multiple elements";

    let arguments = (ins Concrete_BatchLweTensor:$lhs, Concrete_BatchPlaintextTensor:$rhs);
    let results = (outs Concrete_BatchLweTensor:$result);
}

def Concrete_BatchedAddPlaintextLweBufferOp : Concrete_Op<"batched_add_plaintext_lwe_buffer"> {
    let summary = "Batched version of AddPlaintextLweBufferOp, which performs the same operation on multiple elements";

    let arguments = (ins
        Concrete_BatchLweBuffer:$result,
        Concrete_BatchLweBuffer:$lhs,
        Concrete_BatchPlaintextBuffer:$rhs
    );
}

def Concrete_BatchedAddPlaintextCstLweTensorOp : Concrete_Op<"batched_add_plaintext_cst_lwe_tensor", [Pure]> {
    let summary = "Batched version of AddPlaintextLweTensorOp, which performs the same operation on multiple elements";

    let arguments = (ins Concrete_BatchLweTensor:$lhs, I64:$rhs);
    let results = (outs Concrete_BatchLweTensor:$result);
}

def Concrete_BatchedAddPlaintextCstLweBufferOp : Concrete_Op<"batched_add_plaintext_cst_lwe_buffer"> {
    let summary = "Batched version of AddPlaintextLweBufferOp, which performs the same operation on multiple elements";

    let arguments = (ins
        Concrete_BatchLweBuffer:$result,
        Concrete_BatchLweBuffer:$lhs,
        I64:$rhs
    );
}

def Concrete_MulCleartextLweTensorOp : Concrete_Op<"mul_cleartext_lwe_tensor", [Pure]> {
    let summary = "Returns the product of a clear integer and a lwe ciphertext";

    let arguments = (ins Concrete_LweTensor:$lhs, I64:$rhs);
    let results = (outs Concrete_LweTensor:$result);
}

def Concrete_MulCleartextLweBufferOp : Concrete_Op<"mul_cleartext_lwe_buffer"> {
    let summary = "Returns the product of a clear integer and a lwe ciphertext";

    let arguments = (ins
        Concrete_LweBuffer:$result,
        Concrete_LweBuffer:$lhs,
        I64:$rhs
    );
}

def Concrete_BatchedMulCleartextLweTensorOp : Concrete_Op<"batched_mul_cleartext_lwe_tensor", [Pure]> {
    let summary = "Batched version of MulCleartextLweTensorOp, which performs the same operation on multiple elements";

    let arguments = (ins Concrete_BatchLweTensor:$lhs, Concrete_BatchPlaintextTensor:$rhs);
    let results = (outs Concrete_BatchLweTensor:$result);
}

def Concrete_BatchedMulCleartextLweBufferOp : Concrete_Op<"batched_mul_cleartext_lwe_buffer"> {
    let summary = "Batched version of MulCleartextLweBufferOp, which performs the same operation on multiple elements";

    let arguments = (ins
        Concrete_BatchLweBuffer:$result,
        Concrete_BatchLweBuffer:$lhs,
        Concrete_BatchPlaintextBuffer:$rhs
    );
}

def Concrete_BatchedMulCleartextCstLweTensorOp : Concrete_Op<"batched_mul_cleartext_cst_lwe_tensor", [Pure]> {
    let summary = "Batched version of MulCleartextLweTensorOp, which performs the same operation on multiple elements";

    let arguments = (ins Concrete_BatchLweTensor:$lhs, I64:$rhs);
    let results = (outs Concrete_BatchLweTensor:$result);
}

def Concrete_BatchedMulCleartextCstLweBufferOp : Concrete_Op<"batched_mul_cleartext_cst_lwe_buffer"> {
    let summary = "Batched version of MulCleartextLweBufferOp, which performs the same operation on multiple elements";

    let arguments = (ins
        Concrete_BatchLweBuffer:$result,
        Concrete_BatchLweBuffer:$lhs,
        I64:$rhs
    );
}

def Concrete_NegateLweTensorOp : Concrete_Op<"negate_lwe_tensor", [Pure]> {
    let summary = "Negates an lwe ciphertext";

    let arguments = (ins Concrete_LweTensor:$ciphertext);
    let results = (outs Concrete_LweTensor:$result);
}

def Concrete_NegateLweBufferOp : Concrete_Op<"negate_lwe_buffer"> {
    let summary = "Negates an lwe ciphertext";

    let arguments = (ins
        Concrete_LweBuffer:$result,
        Concrete_LweBuffer:$ciphertext
    );
}

def Concrete_BatchedNegateLweTensorOp : Concrete_Op<"batched_negate_lwe_tensor", [Pure]> {
    let summary = "Batched version of NegateLweTensorOp, which performs the same operation on multiple elements";

    let arguments = (ins Concrete_BatchLweTensor:$ciphertext);
    let results = (outs Concrete_BatchLweTensor:$result);
}

def Concrete_BatchedNegateLweBufferOp : Concrete_Op<"batched_negate_lwe_buffer"> {
    let summary = "Batched version of NegateLweBufferOp, which performs the same operation on multiple elements";

    let arguments = (ins
        Concrete_BatchLweBuffer:$result,
        Concrete_BatchLweBuffer:$ciphertext
    );
}

def Concrete_EncodeExpandLutForBootstrapTensorOp : Concrete_Op<"encode_expand_lut_for_bootstrap_tensor", [Pure]> {
    let summary =
    "Encode and expand a lookup table so that it can be used for a bootstrap";

    let arguments = (ins
        Concrete_LutTensor : $input_lookup_table,
        I32Attr: $polySize,
        I32Attr: $outputBits,
        BoolAttr: $isSigned
    );

    let results = (outs Concrete_LutTensor : $result);
}

def Concrete_EncodeExpandLutForBootstrapBufferOp : Concrete_Op<"encode_expand_lut_for_bootstrap_buffer"> {
    let summary =
        "Encode and expand a lookup table so that it can be used for a bootstrap";

    let arguments = (ins
        Concrete_LutBuffer: $result,
        Concrete_LutBuffer: $input_lookup_table,
        I32Attr: $polySize,
        I32Attr: $outputBits,
        BoolAttr : $isSigned
    );
}

def Concrete_EncodeLutForCrtWopPBSTensorOp : Concrete_Op<"encode_lut_for_crt_woppbs_tensor", [Pure]> {
    let summary =
        "Encode and expand a lookup table so that it can be used for a wop pbs";

    let arguments = (ins
        Concrete_LutTensor : $input_lookup_table,
        I64ArrayAttr: $crtDecomposition,
        I64ArrayAttr: $crtBits,
        I32Attr : $modulusProduct,
        BoolAttr: $isSigned
    );

    let results = (outs Concrete_CrtLutsTensor : $result);
}

def Concrete_EncodeLutForCrtWopPBSBufferOp : Concrete_Op<"encode_lut_for_crt_woppbs_buffer"> {
    let summary =
        "Encode and expand a lookup table so that it can be used for a crt wop pbs";

    let arguments = (ins
        Concrete_CrtLutsBuffer : $result,
        Concrete_LutBuffer : $input_lookup_table,
        I64ArrayAttr: $crtDecomposition,
        I64ArrayAttr: $crtBits,
        I32Attr : $modulusProduct,
        BoolAttr: $isSigned
    );
}

def Concrete_EncodePlaintextWithCrtTensorOp : Concrete_Op<"encode_plaintext_with_crt_tensor", [Pure]> {
    let summary =
        "Encodes a plaintext by decomposing it on a crt basis";

    let arguments = (ins
        I64 : $input,
        I64ArrayAttr: $mods,
        I64Attr: $modsProd
    );

    let results = (outs Concrete_CrtPlaintextTensor : $result);
}

def Concrete_EncodePlaintextWithCrtBufferOp : Concrete_Op<"encode_plaintext_with_crt_buffer"> {
    let summary =
        "Encodes a plaintext by decomposing it on a crt basis";

    let arguments = (ins
        Concrete_CrtPlaintextBuffer: $result,
        I64 : $input,
        I64ArrayAttr: $mods,
        I64Attr: $modsProd
    );
}

def Concrete_BootstrapLweTensorOp : Concrete_Op<"bootstrap_lwe_tensor", [Pure]> {
    let summary = "Bootstraps an LWE ciphertext with a GLWE trivial encryption of the lookup table";

    let arguments = (ins
        Concrete_LweTensor:$input_ciphertext,
        Concrete_LweTensor:$lookup_table,
        I32Attr:$inputLweDim,
        I32Attr:$polySize,
        I32Attr:$level,
        I32Attr:$baseLog,
        I32Attr:$glweDimension,
        I32Attr:$bskIndex
    );
    let results = (outs Concrete_LweTensor:$result);
}

def Concrete_BootstrapLweBufferOp : Concrete_Op<"bootstrap_lwe_buffer"> {
    let summary = "Bootstraps a LWE ciphertext with a GLWE trivial encryption of the lookup table";

    let arguments = (ins
        Concrete_LweBuffer:$result,
        Concrete_LweBuffer:$input_ciphertext,
        Concrete_LutBuffer:$lookup_table,
        I32Attr:$inputLweDim,
        I32Attr:$polySize,
        I32Attr:$level,
        I32Attr:$baseLog,
        I32Attr:$glweDimension,
        I32Attr:$bskIndex
    );
}

def Concrete_BatchedBootstrapLweTensorOp : Concrete_Op<"batched_bootstrap_lwe_tensor", [Pure]> {
    let summary = "Batched version of BootstrapLweOp, which performs the same operation on multiple elements";

    let arguments = (ins
        Concrete_BatchLweTensor:$input_ciphertext,
        Concrete_LutTensor:$lookup_table,
        I32Attr:$inputLweDim,
        I32Attr:$polySize,
        I32Attr:$level,
        I32Attr:$baseLog,
        I32Attr:$glweDimension,
        I32Attr:$bskIndex
    );
    let results = (outs Concrete_BatchLweTensor:$result);
}

def Concrete_BatchedBootstrapLweBufferOp : Concrete_Op<"batched_bootstrap_lwe_buffer"> {
    let summary = "Batched version of BootstrapLweOp, which performs the same operation on multiple elements";

    let arguments = (ins
        Concrete_BatchLweBuffer:$result,
        Concrete_BatchLweBuffer:$input_ciphertext,
        Concrete_LutBuffer:$lookup_table,
        I32Attr:$inputLweDim,
        I32Attr:$polySize,
        I32Attr:$level,
        I32Attr:$baseLog,
        I32Attr:$glweDimension,
        I32Attr:$bskIndex
    );
}

def Concrete_BatchedMappedBootstrapLweTensorOp : Concrete_Op<"batched_mapped_bootstrap_lwe_tensor", [Pure]> {
    let summary = "Batched, mapped version of BootstrapLweOp, which performs the same operation on multiple elements";

    let arguments = (ins
        Concrete_BatchLweTensor:$input_ciphertext,
        Concrete_BatchLutTensor:$lookup_table_vector,
        I32Attr:$inputLweDim,
        I32Attr:$polySize,
        I32Attr:$level,
        I32Attr:$baseLog,
        I32Attr:$glweDimension,
        I32Attr:$bskIndex
    );
    let results = (outs Concrete_BatchLweTensor:$result);
}

def Concrete_BatchedMappedBootstrapLweBufferOp : Concrete_Op<"batched_mapped_bootstrap_lwe_buffer"> {
    let summary = "Batched, mapped version of BootstrapLweOp, which performs the same operation on multiple elements";

    let arguments = (ins
        Concrete_BatchLweBuffer:$result,
        Concrete_BatchLweBuffer:$input_ciphertext,
        Concrete_BatchLutBuffer:$lookup_table_vector,
        I32Attr:$inputLweDim,
        I32Attr:$polySize,
        I32Attr:$level,
        I32Attr:$baseLog,
        I32Attr:$glweDimension,
        I32Attr:$bskIndex
    );
}

def Concrete_KeySwitchLweTensorOp : Concrete_Op<"keyswitch_lwe_tensor", [Pure]> {
    let summary = "Performs a keyswitching operation on an LWE ciphertext";

    let arguments = (ins
        // LweKeySwitchKeyType:$keyswitch_key,
        Concrete_LweTensor:$ciphertext,
        I32Attr:$level,
        I32Attr:$baseLog,
        I32Attr:$lwe_dim_in,
        I32Attr:$lwe_dim_out,
        I32Attr:$kskIndex
    );
    let results = (outs Concrete_LweTensor:$result);
}

def Concrete_KeySwitchLweBufferOp : Concrete_Op<"keyswitch_lwe_buffer"> {
    let summary = "Performs a keyswitching operation on an LWE ciphertext";

    let arguments = (ins
        Concrete_LweBuffer:$result,
        Concrete_LweBuffer:$ciphertext,
        I32Attr:$level,
        I32Attr:$baseLog,
        I32Attr:$lwe_dim_in,
        I32Attr:$lwe_dim_out,
        I32Attr:$kskIndex
    );
}

def Concrete_BatchedKeySwitchLweTensorOp : Concrete_Op<"batched_keyswitch_lwe_tensor", [Pure]> {
    let summary = "Batched version of KeySwitchLweOp, which performs the same operation on multiple elements";

    let arguments = (ins
        // LweKeySwitchKeyType:$keyswitch_key,
        Concrete_BatchLweTensor:$ciphertext,
        I32Attr:$level,
        I32Attr:$baseLog,
        I32Attr:$lwe_dim_in,
        I32Attr:$lwe_dim_out,
        I32Attr:$kskIndex
    );
    let results = (outs Concrete_BatchLweTensor:$result);
}

def Concrete_BatchedKeySwitchLweBufferOp : Concrete_Op<"batched_keyswitch_lwe_buffer"> {
    let summary = "Batched version of KeySwitchLweOp, which performs the same operation on multiple elements";

    let arguments = (ins
        Concrete_BatchLweBuffer:$result,
        Concrete_BatchLweBuffer:$ciphertext,
        I32Attr:$level,
        I32Attr:$baseLog,
        I32Attr:$lwe_dim_in,
        I32Attr:$lwe_dim_out,
        I32Attr:$kskIndex
    );
}

def Concrete_WopPBSCRTLweTensorOp : Concrete_Op<"wop_pbs_crt_lwe_tensor", [Pure]> {
    let arguments = (ins
        Concrete_LweCRTTensor:$ciphertext,
        Concrete_CrtLutsTensor:$lookupTable,
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
        I64ArrayAttr:$crtDecomposition,
        // Key indices
        I32Attr:$kskIndex,
        I32Attr:$bskIndex,
        I32Attr:$pkskIndex
    );
    let results = (outs Concrete_LweCRTTensor:$result);
}

def Concrete_WopPBSCRTLweBufferOp : Concrete_Op<"wop_pbs_crt_lwe_buffer"> {
    let arguments = (ins
        Concrete_LweCRTBuffer:$result,
        Concrete_LweCRTBuffer:$ciphertext,
        Concrete_CrtLutsBuffer:$lookup_table,
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
        I64ArrayAttr:$crtDecomposition,
        // Key indices
        I32Attr:$kskIndex,
        I32Attr:$bskIndex,
        I32Attr:$pkskIndex
    );
}

#endif
