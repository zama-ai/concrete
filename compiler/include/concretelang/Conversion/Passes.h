// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.

#ifndef CONCRETELANG_TRANSFORMS_PASSES_H
#define CONCRETELANG_TRANSFORMS_PASSES_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "concretelang/Conversion/HLFHETensorOpsToLinalg/Pass.h"
#include "concretelang/Conversion/HLFHEToMidLFHE/Pass.h"
#include "concretelang/Conversion/LowLFHEToConcreteCAPI/Pass.h"
#include "concretelang/Conversion/LowLFHEUnparametrize/Pass.h"
#include "concretelang/Conversion/MLIRLowerableDialectsToLLVM/Pass.h"
#include "concretelang/Conversion/MidLFHEGlobalParametrization/Pass.h"
#include "concretelang/Conversion/MidLFHEToLowLFHE/Pass.h"
#include "concretelang/Dialect/HLFHE/IR/HLFHEDialect.h"
#include "concretelang/Dialect/LowLFHE/IR/LowLFHEDialect.h"
#include "concretelang/Dialect/MidLFHE/IR/MidLFHEDialect.h"

#define GEN_PASS_CLASSES
#include "concretelang/Conversion/Passes.h.inc"

#endif
