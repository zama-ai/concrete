#ifndef ZAMALANG_DIALECT_BConcrete_IR_BConcrete_OPS
#define ZAMALANG_DIALECT_BConcrete_IR_BConcrete_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Dialect/MemRef/IR/MemRefBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

include "concretelang/Dialect/BConcrete/IR/BConcreteDialect.td"
include "concretelang/Dialect/Concrete/IR/ConcreteTypes.td"
include "concretelang/Dialect/RT/IR/RTDialect.td"
include "concretelang/Dialect/RT/IR/RTTypes.td"

class BConcrete_Op<string mnemonic, list<Trait> traits = []> :
    Op<BConcrete_Dialect, mnemonic, traits>;

// BConcrete tensor operators /////////////////////////////////////////////////

def BConcrete_AddLweTensorOp : BConcrete_Op<"add_lwe_tensor", [NoSideEffect]> {
    let arguments = (ins
        1DTensorOf<[I64]>:$lhs,
        1DTensorOf<[I64]>:$rhs
    );
    let results = (outs 1DTensorOf<[I64]>:$result);
}

def BConcrete_AddPlaintextLweTensorOp : BConcrete_Op<"add_plaintext_lwe_tensor", [NoSideEffect]> {
    let arguments = (ins 1DTensorOf<[I64]>:$lhs, I64:$rhs);
    let results = (outs 1DTensorOf<[I64]>:$result);
}

def BConcrete_MulCleartextLweTensorOp : BConcrete_Op<"mul_cleartext_lwe_tensor", [NoSideEffect]> {
    let arguments = (ins 1DTensorOf<[I64]>:$lhs, I64:$rhs);
    let results = (outs 1DTensorOf<[I64]>:$result);
}

def BConcrete_NegateLweTensorOp : BConcrete_Op<"negate_lwe_tensor", [NoSideEffect]> {
    let arguments = (ins 1DTensorOf<[I64]>:$ciphertext);
    let results = (outs 1DTensorOf<[I64]>:$result);
}

def BConcrete_KeySwitchLweTensorOp : BConcrete_Op<"keyswitch_lwe_tensor", [NoSideEffect]> {
    let arguments = (ins
        // LweKeySwitchKeyType:$keyswitch_key,
        1DTensorOf<[I64]>:$ciphertext,
        I32Attr:$level,
        I32Attr:$baseLog,
        I32Attr:$lwe_dim_in,
        I32Attr:$lwe_dim_out
    );
    let results = (outs 1DTensorOf<[I64]>:$result);
}

def BConcrete_BatchedKeySwitchLweTensorOp : BConcrete_Op<"batched_keyswitch_lwe_tensor", [NoSideEffect]> {
    let arguments = (ins
        // LweKeySwitchKeyType:$keyswitch_key,
        2DTensorOf<[I64]>:$ciphertext,
        I32Attr:$level,
        I32Attr:$baseLog,
        I32Attr:$lwe_dim_in,
        I32Attr:$lwe_dim_out
    );
    let results = (outs 2DTensorOf<[I64]>:$result);
}

def BConcrete_EncodeExpandLutForBootstrapTensorOp : BConcrete_Op<"encode_expand_lut_for_bootstrap_tensor", [NoSideEffect]> {
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

def BConcrete_EncodeExpandLutForWopPBSTensorOp : BConcrete_Op<"encode_expand_lut_for_woppbs_tensor", [NoSideEffect]> {
    let summary =
        "Encode and expand a lookup table so that it can be used for a wop pbs.";

    let arguments = (ins
        1DTensorOf<[I64]> : $input_lookup_table,
        I64ArrayAttr: $crtDecomposition,
        I64ArrayAttr: $crtBits,
        I32Attr : $polySize,
        I32Attr : $modulusProduct,
        BoolAttr: $isSigned
    );

    let results = (outs 1DTensorOf<[I64]> : $result);
}


def BConcrete_EncodePlaintextWithCrtTensorOp : BConcrete_Op<"encode_plaintext_with_crt_tensor", [NoSideEffect]> {
    let summary =
        "Encodes a plaintext by decomposing it on a crt basis.";

    let arguments = (ins
        I64 : $input,
        I64ArrayAttr: $mods,
        I64Attr: $modsProd
    );

    let results = (outs 1DTensorOf<[I64]> : $result);
}


def BConcrete_BootstrapLweTensorOp : BConcrete_Op<"bootstrap_lwe_tensor", [NoSideEffect]> {
    let arguments = (ins
        1DTensorOf<[I64]>:$input_ciphertext,
        1DTensorOf<[I64]>:$lookup_table,
        I32Attr:$inputLweDim,
        I32Attr:$polySize,
        I32Attr:$level,
        I32Attr:$baseLog,
        I32Attr:$glweDimension,
        I32Attr:$outPrecision
    );
    let results = (outs 1DTensorOf<[I64]>:$result);
}

def BConcrete_BatchedBootstrapLweTensorOp : BConcrete_Op<"batched_bootstrap_lwe_tensor", [NoSideEffect]> {
    let arguments = (ins
        2DTensorOf<[I64]>:$input_ciphertext,
        1DTensorOf<[I64]>:$lookup_table,
        I32Attr:$inputLweDim,
        I32Attr:$polySize,
        I32Attr:$level,
        I32Attr:$baseLog,
        I32Attr:$glweDimension,
        I32Attr:$outPrecision
    );
    let results = (outs 2DTensorOf<[I64]>:$result);
}

def BConcrete_WopPBSCRTLweTensorOp : BConcrete_Op<"wop_pbs_crt_lwe_tensor", [NoSideEffect]> {
    let arguments = (ins
        2DTensorOf<[I64]>:$ciphertext,
        1DTensorOf<[I64]>:$lookupTable,
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
        I32Attr : $circuitBootstrapBaseLog
    );
    let results = (outs 2DTensorOf<[I64]>:$result);
}

// BConcrete memref operators /////////////////////////////////////////////////

def BConcrete_LweBuffer : MemRefRankOf<[I64], [1]>;
def BConcrete_LutBuffer : MemRefRankOf<[I64], [1]>;
def BConcrete_CrtPlaintextBuffer : MemRefRankOf<[I64], [1]>;
def BConcrete_LweCRTBuffer : MemRefRankOf<[I64], [2]>;
def BConcrete_BatchLweBuffer : MemRefRankOf<[I64], [2]>;

def BConcrete_AddLweBufferOp : BConcrete_Op<"add_lwe_buffer"> {
    let arguments = (ins
        BConcrete_LweBuffer:$result,
        BConcrete_LweBuffer:$lhs,
        BConcrete_LweBuffer:$rhs
    );
}

def BConcrete_AddPlaintextLweBufferOp : BConcrete_Op<"add_plaintext_lwe_buffer"> {
    let arguments = (ins
        BConcrete_LweBuffer:$result,
        BConcrete_LweBuffer:$lhs,
        I64:$rhs
    );
}

def BConcrete_MulCleartextLweBufferOp : BConcrete_Op<"mul_cleartext_lwe_buffer"> {
    let arguments = (ins
        BConcrete_LweBuffer:$result,
        BConcrete_LweBuffer:$lhs,
        I64:$rhs
    );
}

def BConcrete_NegateLweBufferOp : BConcrete_Op<"negate_lwe_buffer"> {
    let arguments = (ins
        BConcrete_LweBuffer:$result,
        BConcrete_LweBuffer:$ciphertext
    );
}

def BConcrete_KeySwitchLweBufferOp : BConcrete_Op<"keyswitch_lwe_buffer"> {
    let arguments = (ins
        BConcrete_LweBuffer:$result,
        BConcrete_LweBuffer:$ciphertext,
        I32Attr:$level,
        I32Attr:$baseLog,
        I32Attr:$lwe_dim_in,
        I32Attr:$lwe_dim_out
    );
}

def BConcrete_BatchedKeySwitchLweBufferOp : BConcrete_Op<"batched_keyswitch_lwe_buffer"> {
    let arguments = (ins
        BConcrete_BatchLweBuffer:$result,
        BConcrete_BatchLweBuffer:$ciphertext,
        I32Attr:$level,
        I32Attr:$baseLog,
        I32Attr:$lwe_dim_in,
        I32Attr:$lwe_dim_out
    );
}

def BConcrete_EncodeExpandLutForBootstrapBufferOp : BConcrete_Op<"encode_expand_lut_for_bootstrap_buffer"> {
    let summary =
        "Encode and expand a lookup table so that it can be used for a bootstrap.";

    let arguments = (ins
        BConcrete_LutBuffer: $result,
        BConcrete_LutBuffer: $input_lookup_table,
        I32Attr: $polySize,
        I32Attr: $outputBits,
        BoolAttr : $isSigned
    );
}

def BConcrete_EncodeExpandLutForWopPBSBufferOp : BConcrete_Op<"encode_expand_lut_for_woppbs_buffer"> {
    let summary =
        "Encode and expand a lookup table so that it can be used for a wop pbs.";

    let arguments = (ins
        BConcrete_LutBuffer : $result,
        BConcrete_LutBuffer : $input_lookup_table,
        I64ArrayAttr: $crtDecomposition,
        I64ArrayAttr: $crtBits,
        I32Attr : $polySize,
        I32Attr : $modulusProduct,
        BoolAttr: $isSigned
    );
}

def BConcrete_EncodePlaintextWithCrtBufferOp : BConcrete_Op<"encode_plaintext_with_crt_buffer"> {
    let summary =
        "Encodes a plaintext by decomposing it on a crt basis.";

    let arguments = (ins
        BConcrete_CrtPlaintextBuffer: $result,
        I64 : $input,
        I64ArrayAttr: $mods,
        I64Attr: $modsProd
    );
}

def BConcrete_BootstrapLweBufferOp : BConcrete_Op<"bootstrap_lwe_buffer"> {
    let arguments = (ins
        BConcrete_LweBuffer:$result,
        BConcrete_LweBuffer:$input_ciphertext,
        BConcrete_LutBuffer:$lookup_table,
        I32Attr:$inputLweDim,
        I32Attr:$polySize,
        I32Attr:$level,
        I32Attr:$baseLog,
        I32Attr:$glweDimension,
        I32Attr:$outPrecision
    );
}

def BConcrete_BatchedBootstrapLweBufferOp : BConcrete_Op<"batched_bootstrap_lwe_buffer"> {
    let arguments = (ins
        BConcrete_BatchLweBuffer:$result,
        BConcrete_BatchLweBuffer:$input_ciphertext,
        BConcrete_LutBuffer:$lookup_table,
        I32Attr:$inputLweDim,
        I32Attr:$polySize,
        I32Attr:$level,
        I32Attr:$baseLog,
        I32Attr:$glweDimension,
        I32Attr:$outPrecision
    );
}

def BConcrete_WopPBSCRTLweBufferOp : BConcrete_Op<"wop_pbs_crt_lwe_buffer"> {
    let arguments = (ins
        BConcrete_LweCRTBuffer:$result,
        BConcrete_LweCRTBuffer:$ciphertext,
        BConcrete_LutBuffer:$lookup_table,
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
        I64ArrayAttr:$crtDecomposition
    );
}

#endif
