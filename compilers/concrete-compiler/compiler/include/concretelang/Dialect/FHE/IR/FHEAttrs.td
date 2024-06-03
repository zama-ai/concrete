//===- FHEAttrs.td - FHE dialect attrs ----------------*- tablegen
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONCRETELANG_DIALECT_FHE_IR_FHE_ATTRS
#define CONCRETELANG_DIALECT_FHE_IR_FHE_ATTRS

include "concretelang/Dialect/FHE/IR/FHEDialect.td"
include "concretelang/Dialect/FHE/IR/FHETypes.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

class FHE_Attr<string name, string attrMnemonic, list<Trait> traits = []>: AttrDef<FHE_Dialect, name, traits> {
    let mnemonic = attrMnemonic;
}

def FHE_PartitionAttr: FHE_Attr<"Partition", "partition"> {

    let summary = "An attribute representing a partition.";

    let parameters = (ins
        "int":$lweDim,
        "int":$glweDim,
        "int":$polySize,
        "FloatAttr":$lweNoiseDistrStdDev,
        "FloatAttr":$glweNoiseDistrStdDev,
        "int":$pbsBaseLog,
        "int":$pbsLevel,
        "int":$ksBaseLog,
        "int":$ksLevel,
        "int":$msgModulus,
        "int":$carryModulus,
        "int":$maxNoiseLevel,
        "FloatAttr":$log2PFail,
        "bool": $bigEncryptionKey,
        DefaultValuedParameter<"int", "-1">: $ciphertextModulus
    );

    let assemblyFormat = "`<` `lwe_dim` $lweDim `,` `glwe_dim` $glweDim `,` `poly_size` $polySize `,` `lwe_noise_std` $lweNoiseDistrStdDev `,` `glwe_noise_std` $glweNoiseDistrStdDev `,` `pbs_base_log` $pbsBaseLog `,` `pbs_level` $pbsLevel `,` `ks_base_log` $ksBaseLog `,` `ks_level` $ksLevel `,` `msg_modulus` $msgModulus `,` `carry_modulus` $carryModulus `,` `max_noise_level` $maxNoiseLevel `,` `log2_pfail` $log2PFail `,` `big_key` $bigEncryptionKey `,` `ct_modulus` $ciphertextModulus `>`";
}

#endif
