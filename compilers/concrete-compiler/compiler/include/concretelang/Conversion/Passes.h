// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_TRANSFORMS_PASSES_H
#define CONCRETELANG_TRANSFORMS_PASSES_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "concretelang/Conversion/ConcreteToCAPI/Pass.h"
#include "concretelang/Conversion/ExtractSDFGOps/Pass.h"
#include "concretelang/Conversion/FHETensorOpsToLinalg/Pass.h"
#include "concretelang/Conversion/FHEToTFHECrt/Pass.h"
#include "concretelang/Conversion/FHEToTFHEScalar/Pass.h"
#include "concretelang/Conversion/LinalgExtras/Passes.h"
#include "concretelang/Conversion/MLIRLowerableDialectsToLLVM/Pass.h"
#include "concretelang/Conversion/SDFGToStreamEmulator/Pass.h"
#include "concretelang/Conversion/SimulateTFHE/Pass.h"
#include "concretelang/Conversion/TFHEGlobalParametrization/Pass.h"
#include "concretelang/Conversion/TFHEKeyNormalization/Pass.h"
#include "concretelang/Conversion/TFHEToConcrete/Pass.h"
#include "concretelang/Conversion/TracingToCAPI/Pass.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteDialect.h"
#include "concretelang/Dialect/FHE/IR/FHEDialect.h"
#include "concretelang/Dialect/SDFG/IR/SDFGDialect.h"
#include "concretelang/Dialect/TFHE/IR/TFHEDialect.h"
#include "concretelang/Dialect/Tracing/IR/TracingDialect.h"

#define GEN_PASS_CLASSES
#include "concretelang/Conversion/Passes.h.inc"

#endif
