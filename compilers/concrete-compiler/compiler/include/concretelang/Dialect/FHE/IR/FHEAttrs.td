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
        "StringAttr":$name,
        "uint64_t":$lweDim,
        "uint64_t":$glweDim,
        "uint64_t":$polySize,
        "uint64_t":$pbsBaseLog,
        "uint64_t":$pbsLevel
    );
    let assemblyFormat = "`<` `name` $name `,` `lwe_dim` $lweDim `,` `glwe_dim` $glweDim `,` `poly_size` $polySize `,` `pbs_base_log` $pbsBaseLog `,` `pbs_level` $pbsLevel `>`";
}

#endif
