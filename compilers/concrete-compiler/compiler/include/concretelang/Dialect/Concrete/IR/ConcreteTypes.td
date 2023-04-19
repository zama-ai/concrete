//===- ConcreteTypes.td - Concrete dialect types ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONCRETELANG_DIALECT_Concrete_IR_Concrete_TYPES
#define CONCRETELANG_DIALECT_Concrete_IR_Concrete_TYPES

include "mlir/IR/BuiltinTypes.td"

include "concretelang/Dialect/Concrete/IR/ConcreteDialect.td"

class Concrete_Type<string name, list<Trait> traits = []> : TypeDef<Concrete_Dialect, name, traits> { }

def Concrete_Context : Concrete_Type<"Context"> {
    let mnemonic = "context";

    let summary = "A runtime context";

    let description = [{
       An abstract runtime context to pass contextual value, like public keys, ...
    }];
}

#endif
