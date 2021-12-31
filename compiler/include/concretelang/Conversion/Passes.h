// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license
// information.

#ifndef CONCRETELANG_TRANSFORMS_PASSES_H
#define CONCRETELANG_TRANSFORMS_PASSES_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "concretelang/Conversion/ConcreteToConcreteCAPI/Pass.h"
#include "concretelang/Conversion/ConcreteUnparametrize/Pass.h"
#include "concretelang/Conversion/FHETensorOpsToLinalg/Pass.h"
#include "concretelang/Conversion/FHEToTFHE/Pass.h"
#include "concretelang/Conversion/MLIRLowerableDialectsToLLVM/Pass.h"
#include "concretelang/Conversion/TFHEGlobalParametrization/Pass.h"
#include "concretelang/Conversion/TFHEToConcrete/Pass.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteDialect.h"
#include "concretelang/Dialect/FHE/IR/FHEDialect.h"
#include "concretelang/Dialect/TFHE/IR/TFHEDialect.h"

#define GEN_PASS_CLASSES
#include "concretelang/Conversion/Passes.h.inc"

#endif
