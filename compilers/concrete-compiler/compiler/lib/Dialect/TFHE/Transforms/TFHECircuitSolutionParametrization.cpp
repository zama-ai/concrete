// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "concretelang/Dialect/TFHE/IR/TFHEOps.h"
#include "concretelang/Dialect/TFHE/Transforms/Transforms.h"
#include "concretelang/Support/Constants.h"
#include "concretelang/Support/logging.h"

namespace mlir {
namespace concretelang {

namespace {

#define DEBUG(MSG)                                                             \
  if (llvm::DebugFlag)                                                         \
    llvm::errs() << MSG << "\n";

#define VERBOSE(MSG)                                                           \
  if (mlir::concretelang::isVerbose()) {                                       \
    llvm::errs() << MSG << "\n";                                               \
  }

namespace TFHE = mlir::concretelang::TFHE;

/// Optimization pass that should choose more efficient ways of performing
/// crypto operations.
class TFHECircuitSolutionParametrizationPass
    : public TFHECircuitSolutionParametrizationBase<
          TFHECircuitSolutionParametrizationPass> {
public:
  TFHECircuitSolutionParametrizationPass(
      concrete_optimizer::dag::CircuitSolution solution)
      : solution(solution){};

  void runOnOperation() override {
    mlir::Operation *op = getOperation();
    op->walk([&](mlir::func::FuncOp func) {
      DEBUG("apply solution: \n" << solution.dump().c_str());
      DEBUG("process func: " << func);
      // Process function arguments, change type of arguments according of the
      // optimizer identifier stored in the "TFHE.OId" attribute.
      for (size_t i = 0; i < func.getNumArguments(); i++) {
        auto arg = func.getArgument(i);
        auto attr = func.getArgAttrOfType<mlir::IntegerAttr>(i, "TFHE.OId");
        if (attr != nullptr) {
          DEBUG("process arg = " << arg)
          arg.setType(getParametrizedType(arg.getType(), attr));
        } else {
          DEBUG("skip arg " << arg)
        }
      }
      // Process operations, apply the instructions keys according of the
      // optimizer identifier stored in the "TFHE.OId"
      VERBOSE("\n### BEFORE Apply instruction keys " << func);
      applyConversionKeys(func);
      // The keyswitch operator is an internal node of the optimizer tlu node,
      // so it don't follow the same rule than the other operator on the type of
      // outputs
      VERBOSE("\n### BEFORE Fixup keys \n" << func);
      fixupKeyswitchOuputs(func);
      // Propagate types on non parametrized operators
      VERBOSE("\n### BEFORE Fixup non parametrized ops \n" << func);
      fixupNonParametrizedOps(func);
      // Fixup incompatible operators with extra conversion keys
      VERBOSE("\n### BEFORE Fixup with extra conversion keys \n" << func);
      fixupIncompatibleLeveledOpWithExtraConversionKeys(func);
      VERBOSE("\n### BEFORE Fixup non parametrized producer \n" << func);
      fixupNonParametrizedProducer(func);
      // Fixup the function signature
      VERBOSE("\n### BEFORE Fixup function signature \n" << func);
      fixupFunctionSignature(func);
      // Remove optimizer identifiers
      VERBOSE("\n### BEFORE Remove optimizer identifiers \n" << func);
      removeOptimizerIdentifiers(func);
    });
  }

  static mlir::Type getParametrizedType(mlir::Type originalType,
                                        TFHE::GLWECipherTextType newGlwe) {
    if (auto oldGlwe = originalType.dyn_cast<TFHE::GLWECipherTextType>();
        oldGlwe != nullptr) {
      assert(oldGlwe.getKey().isNone());
      return newGlwe;
    } else if (auto oldTensor = originalType.dyn_cast<mlir::RankedTensorType>();
               oldTensor != nullptr) {
      auto oldGlwe =
          oldTensor.getElementType().dyn_cast<TFHE::GLWECipherTextType>();
      assert(oldGlwe != nullptr);
      assert(oldGlwe.getKey().isNone());
      return mlir::RankedTensorType::get(oldTensor.getShape(), newGlwe);
    }
    assert(false);
  }

  mlir::Type getParametrizedType(mlir::Type originalType,
                                 mlir::IntegerAttr optimizerAttrID) {
    auto context = originalType.getContext();
    auto newGlwe =
        getOutputLWECipherTextType(context, optimizerAttrID.getInt());
    return getParametrizedType(originalType, newGlwe);
  }

  static TFHE::GLWECipherTextType getGlweTypeFromType(mlir::Type type) {
    if (auto glwe = type.dyn_cast<TFHE::GLWECipherTextType>();
        glwe != nullptr) {
      return glwe;
    } else if (auto tensor = type.dyn_cast<mlir::RankedTensorType>();
               tensor != nullptr) {
      auto glwe = tensor.getElementType().dyn_cast<TFHE::GLWECipherTextType>();
      if (glwe == nullptr) {
        return nullptr;
      }
      return glwe;
    }
    return nullptr;
  }

  // Return the
  static TFHE::GLWECipherTextType
  getParametrizedGlweTypeFromType(mlir::Type type) {
    auto glwe = getGlweTypeFromType(type);
    if (glwe != nullptr && glwe.getKey().isParameterized()) {
      return glwe;
    }
    return nullptr;
  }

  // Returns true if the type is or contains a glwe type with a none key.
  static bool isNoneGlweType(mlir::Type type) {
    auto glwe = getGlweTypeFromType(type);
    return glwe != nullptr && glwe.getKey().isNone();
  }

  void applyConversionKeys(mlir::func::FuncOp func) {
    auto context = func.getContext();
    func.walk([&](mlir::Operation *op) {
      auto attrOptimizerID = op->getAttrOfType<IntegerAttr>("TFHE.OId");
      // Skip operation is no optimizer identifier
      if (attrOptimizerID == nullptr) {
        DEBUG("skip operation: " << op->getName())
        return;
      }
      DEBUG("process operation: " << *op);
      auto optimizerID = attrOptimizerID.getInt();
      // Change the output type of the operation
      for (auto result : op->getResults()) {
        result.setType(getParametrizedType(result.getType(), attrOptimizerID));
      }
      // Set the keyswitch_key attribute
      // TODO: Change ambiguous attribute name
      auto attrKeyswitchKey =
          op->getAttrOfType<TFHE::GLWEKeyswitchKeyAttr>("key");
      if (attrKeyswitchKey == nullptr) {
        DEBUG("no keyswitch key");
      } else {
        op->setAttr("key", getKeyswitchKeyAttr(context, optimizerID));
      }
      // Set boostrap_key attribute
      // TODO: Change ambiguous attribute name
      auto attrBootstrapKey =
          op->getAttrOfType<TFHE::GLWEBootstrapKeyAttr>("key");
      if (attrBootstrapKey == nullptr) {
        DEBUG("no bootstrap key");
      } else {
        op->setAttr("key", getBootstrapKeyAttr(context, optimizerID));
      }
    });
  }

  void fixupKeyswitchOuputs(mlir::func::FuncOp func) {
    auto context = func.getContext();
    func.walk([&](TFHE::KeySwitchGLWEOp op) {
      DEBUG("process op: " << op)
      auto attrKeyswitchKey =
          op->getAttrOfType<TFHE::GLWEKeyswitchKeyAttr>("key");
      assert(attrKeyswitchKey != nullptr);
      auto outputKey = attrKeyswitchKey.getOutputKey();
      outputKey = GLWESecretKeyAsLWE(outputKey);
      op.getResult().setType(TFHE::GLWECipherTextType::get(context, outputKey));
      DEBUG("fixed op: " << op)
    });
    // Fixup input of the boostrap operator
    DEBUG("### Fixup input tlu of bootstrap")
    func.walk([&](TFHE::BootstrapGLWEOp op) {
      DEBUG("process op: " << op)
      auto attrBootstrapKey =
          op->getAttrOfType<TFHE::GLWEBootstrapKeyAttr>("key");
      assert(attrBootstrapKey != nullptr);
      auto polySize = attrBootstrapKey.getPolySize();
      auto lutDefiningOp = op.getLookupTable().getDefiningOp();
      // Dirty fixup of the lookup table as we known the operators that can
      // define it
      // TODO: Do something more robust, using the GLWE type?
      mlir::Builder builder(op->getContext());
      assert(lutDefiningOp != nullptr);
      if (auto encodeOp = mlir::dyn_cast<TFHE::EncodeExpandLutForBootstrapOp>(
              lutDefiningOp);
          encodeOp != nullptr) {
        encodeOp.setPolySize(polySize);
      } else if (auto constantOp =
                     mlir::dyn_cast<arith::ConstantOp>(lutDefiningOp)) {
        // Rounded PBS case
        auto denseAttr =
            constantOp.getValueAttr().dyn_cast<mlir::DenseIntElementsAttr>();
        auto val = denseAttr.getValues<int64_t>()[0];
        std::vector<int64_t> lut(polySize, val);
        constantOp.setValueAttr(mlir::DenseIntElementsAttr::get(
            mlir::RankedTensorType::get(lut.size(), builder.getIntegerType(64)),
            lut));
      }
      op.getLookupTable().setType(mlir::RankedTensorType::get(
          mlir::ArrayRef<int64_t>(polySize), builder.getI64Type()));
      // Also fixup the bootstrap key as the TFHENormalization rely on
      // GLWESecretKey structure and not on identifier
      // TODO: FIXME
      auto outputKey = attrBootstrapKey.getOutputKey().getParameterized();
      auto newOutputKey = TFHE::GLWESecretKey::newParameterized(
          outputKey->polySize * outputKey->dimension, 1, outputKey->identifier);
      auto newAttrBootstrapKey = TFHE::GLWEBootstrapKeyAttr::get(
          context, attrBootstrapKey.getInputKey(), newOutputKey,
          attrBootstrapKey.getPolySize(), attrBootstrapKey.getGlweDim(),
          attrBootstrapKey.getLevels(), attrBootstrapKey.getBaseLog(), -1);
      op.setKeyAttr(newAttrBootstrapKey);
    });
  }

  static void
  fixupNonParametrizedOp(mlir::Operation *op,
                         TFHE::GLWECipherTextType parametrizedGlweType) {
    DEBUG("  START Fixup {" << *op)
    for (auto result : op->getResults()) {
      if (isNoneGlweType(result.getType())) {
        result.setType(
            getParametrizedType(result.getType(), parametrizedGlweType));
        DEBUG("      -> Fixed result " << result)
        // Recurse on all users of the fixed result
        for (auto user : result.getUsers()) {
          DEBUG("    -> Propagate on user " << *user)
          fixupNonParametrizedOp(user, parametrizedGlweType);
        }
      }
    }
    // Recursively fixup producer of op operands
    mlir::Block *parentBlock = nullptr;
    for (auto operand : op->getOperands()) {
      if (isNoneGlweType(operand.getType())) {
        if (auto blockArg = operand.dyn_cast<mlir::BlockArgument>();
            blockArg != nullptr) {
          DEBUG("    -> Fixing block arg " << blockArg)
          blockArg.setType(
              getParametrizedType(blockArg.getType(), parametrizedGlweType));
          for (auto users : blockArg.getUsers()) {
            fixupNonParametrizedOp(users, parametrizedGlweType);
          }
          auto blockOwner = blockArg.getOwner();
          if (blockOwner->isEntryBlock()) {
            DEBUG("      -> Will propagate on parent op "
                  << blockOwner->getParentOp());
            assert(parentBlock == blockOwner || parentBlock == nullptr);
            parentBlock = blockOwner;
          }
          continue;
        }
      }
    }
    DEBUG("  } END Fixup")
    if (parentBlock != nullptr) {
      fixupNonParametrizedOp(parentBlock->getParentOp(), parametrizedGlweType);
    }
  }

  static void fixupNonParametrizedOps(mlir::func::FuncOp func) {
    // Lookup all operators that uses function arguments
    for (const auto arg : func.getArguments()) {
      auto parametrizedGlweType =
          getParametrizedGlweTypeFromType(arg.getType());
      if (parametrizedGlweType != nullptr) {
        DEBUG("  -> Fixup uses of arg " << arg)
        // The argument is glwe, so propagate the glwe parametrization to all
        // operators which use it
        for (auto userOp : arg.getUsers()) {
          fixupNonParametrizedOp(userOp, parametrizedGlweType);
        }
      }
    }
    // Fixup all operators that take at least a parametrized glwe and produce an
    // none glwe
    func.walk([&](mlir::Operation *op) {
      for (auto operand : op->getOperands()) {
        auto parametrizedGlweType =
            getParametrizedGlweTypeFromType(operand.getType());
        if (parametrizedGlweType != nullptr) {
          // An operand is a parametrized glwe
          for (auto result : op->getResults()) {
            if (isNoneGlweType(result.getType())) {
              DEBUG("  -> Fixup illegal op " << *op)
              fixupNonParametrizedOp(op, parametrizedGlweType);
              return;
            }
          }
        }
      }
    });
  }

  // Some of TFHE.glwe producer may not be a TFHE operators (like the tensor
  // allocation), this pass is use to propagate parametrized type on those kind
  // of
  void fixupNonParametrizedProducer(mlir::func::FuncOp func) {
    func.walk([&](mlir::Operation *op) {
      for (auto result : op->getResults()) {
        auto parametrizedGlweType =
            getParametrizedGlweTypeFromType(result.getType());
        if (parametrizedGlweType == nullptr)
          continue;
        for (auto operand : op->getOperands()) {
          auto glwe = getGlweTypeFromType(operand.getType());
          if (glwe != nullptr && glwe.getKey().isNone()) {
            operand.setType(
                getParametrizedType(operand.getType(), parametrizedGlweType));
          }
        }
      }
    });
  }

  void
  fixupIncompatibleLeveledOpWithExtraConversionKeys(mlir::func::FuncOp func) {
    auto context = func.getContext();
    func.walk([&](mlir::Operation *op) {
      // Skip bootstrap/keyswitch
      if (mlir::isa<TFHE::BootstrapGLWEOp>(op) ||
          mlir::isa<TFHE::KeySwitchGLWEOp>(op)) {
        return;
      }
      auto attrOptimizerID = op->getAttrOfType<IntegerAttr>("TFHE.OId");
      // Skip operation with no optimizer identifier
      if (attrOptimizerID == nullptr) {
        return;
      }
      DEBUG("  -> process op: " << *op)
      // TFHE operators have only one ciphertext result
      assert(op->getNumResults() == 1);
      auto resType =
          op->getResult(0).getType().dyn_cast<TFHE::GLWECipherTextType>();
      // For each ciphertext operands apply the extra keyswitch if found
      for (const auto &p : llvm::enumerate(op->getOperands())) {
        if (resType == nullptr) {
          // We don't expect tensor operands to exist at this point of the
          // pipeline for now, but if we happen to have some, this assert
          // will break, and things will need to be changed to allow tensor
          // ops to be parameterized.
          // TODO: Actually this case could happens with tensor manipulation
          // operators, so for now we just skip it and that should be fixed
          // and tested. As the operand will not be fixed the validation of
          // operators should not validate the operators.
          continue;
        }
        auto operand = p.value();
        auto operandIdx = p.index();
        DEBUG("    -> processing operand " << operand);
        auto operandType =
            operand.getType().dyn_cast<TFHE::GLWECipherTextType>();
        if (operandType == nullptr) {
          DEBUG("      -> skip operand, no glwe");
          continue;
        }
        if (operandType.getKey() == resType.getKey()) {
          DEBUG("      -> skip operand, unnecessary conversion");
          continue;
        }
        // Lookup for the extra conversion key
        DEBUG("      -> get extra conversion key")
        auto extraConvKey = getExtraConversionKeyAttr(
            context, operandType.getKey(), resType.getKey());
        if (extraConvKey == nullptr) {
          DEBUG("      -> extra conversion key, not found")
          assert(false);
        }
        mlir::IRRewriter rewriter(context);
        rewriter.setInsertionPoint(op);
        auto newKSK = rewriter.create<TFHE::KeySwitchGLWEOp>(
            op->getLoc(), resType, operand, extraConvKey);
        DEBUG("create extra conversion keyswitch: " << newKSK);
        op->setOperand(operandIdx, newKSK);
      }
    });
  }

  static void removeOptimizerIdentifiers(mlir::func::FuncOp func) {
    for (size_t i = 0; i < func.getNumArguments(); i++) {
      func.removeArgAttr(i, "TFHE.OId");
    }
    func.walk([&](mlir::Operation *op) { op->removeAttr("TFHE.OId"); });
  }

  static void fixupFunctionSignature(mlir::func::FuncOp func) {
    mlir::SmallVector<mlir::Type> inputs;
    mlir::SmallVector<mlir::Type> outputs;
    // Set inputs by looking actual arguments types
    for (auto arg : func.getArguments()) {
      inputs.push_back(arg.getType());
    }
    // Look for return to set the outputs
    func.walk([&](mlir::func::ReturnOp returnOp) {
      // TODO: multiple return op
      for (auto output : returnOp->getOperandTypes()) {
        outputs.push_back(output);
      }
    });
    auto funcType =
        mlir::FunctionType::get(func->getContext(), inputs, outputs);
    func.setFunctionType(funcType);
  }

  const concrete_optimizer::dag::InstructionKeys &
  getInstructionKey(size_t optimizerID) {
    DEBUG("lookup instruction key: #" << optimizerID);
    return solution.instructions_keys[optimizerID];
  }

  const TFHE::GLWESecretKey GLWESecretKeyAsLWE(TFHE::GLWESecretKey key) {
    auto keyP = key.getParameterized();
    assert(keyP.has_value());
    return TFHE::GLWESecretKey::newParameterized(
        keyP->polySize * keyP->dimension, 1, keyP->identifier);
  }

  const TFHE::GLWESecretKey
  toGLWESecretKey(concrete_optimizer::dag::SecretLweKey key) {
    return TFHE::GLWESecretKey::newParameterized(
        key.glwe_dimension, key.polynomial_size, key.identifier);
  }

  const TFHE::GLWESecretKey
  toLWESecretKey(concrete_optimizer::dag::SecretLweKey key) {
    return TFHE::GLWESecretKey::newParameterized(
        key.glwe_dimension * key.polynomial_size, 1, key.identifier);
  }

  const TFHE::GLWESecretKey getLWESecretKey(size_t keyID) {
    DEBUG("lookup secret key: #" << keyID);
    auto key = solution.circuit_keys.secret_keys[keyID];
    assert(keyID == key.identifier);
    return toLWESecretKey(key);
  }

  const TFHE::GLWESecretKey getInputLWESecretKey(size_t optimizerID) {
    auto keyID = getInstructionKey(optimizerID).input_key;
    return getLWESecretKey(keyID);
  }

  const TFHE::GLWESecretKey getOutputLWESecretKey(size_t optimizerID) {
    auto keyID = getInstructionKey(optimizerID).output_key;
    return getLWESecretKey(keyID);
  }

  const TFHE::GLWEKeyswitchKeyAttr
  getKeyswitchKeyAttr(mlir::MLIRContext *context, size_t optimizerID) {
    auto keyID = getInstructionKey(optimizerID).tlu_keyswitch_key;
    DEBUG("lookup keyswicth key: #" << keyID);
    auto key = solution.circuit_keys.keyswitch_keys[keyID];
    return TFHE::GLWEKeyswitchKeyAttr::get(
        context, toLWESecretKey(key.input_key), toLWESecretKey(key.output_key),
        key.ks_decomposition_parameter.level,
        key.ks_decomposition_parameter.log2_base, -1);
  }

  const TFHE::GLWEKeyswitchKeyAttr
  getExtraConversionKeyAttr(mlir::MLIRContext *context,
                            TFHE::GLWESecretKey inputKey,
                            TFHE::GLWESecretKey ouputKey) {
    auto convKSK = std::find_if(
        solution.circuit_keys.conversion_keyswitch_keys.begin(),
        solution.circuit_keys.conversion_keyswitch_keys.end(),
        [&](concrete_optimizer::dag::ConversionKeySwitchKey &arg) {
          assert(ouputKey.isParameterized() && inputKey.isParameterized());
          return arg.input_key.identifier ==
                     inputKey.getParameterized()->identifier &&
                 arg.output_key.identifier ==
                     ouputKey.getParameterized()->identifier;
        });
    assert(convKSK != solution.circuit_keys.conversion_keyswitch_keys.end());
    return TFHE::GLWEKeyswitchKeyAttr::get(
        context, toLWESecretKey(convKSK->input_key),
        toLWESecretKey(convKSK->output_key),
        convKSK->ks_decomposition_parameter.level,
        convKSK->ks_decomposition_parameter.log2_base, -1);
  }

  const TFHE::GLWEBootstrapKeyAttr
  getBootstrapKeyAttr(mlir::MLIRContext *context, size_t optimizerID) {
    auto keyID = getInstructionKey(optimizerID).tlu_bootstrap_key;
    DEBUG("lookup bootstrap key: #" << keyID);
    auto key = solution.circuit_keys.bootstrap_keys[keyID];
    return TFHE::GLWEBootstrapKeyAttr::get(
        context, toLWESecretKey(key.input_key), toGLWESecretKey(key.output_key),
        key.output_key.polynomial_size, key.output_key.glwe_dimension,
        key.br_decomposition_parameter.level,
        key.br_decomposition_parameter.log2_base, -1);
  }

  const TFHE::GLWECipherTextType
  getOutputLWECipherTextType(mlir::MLIRContext *context, size_t optimizerID) {
    auto outputKey = getOutputLWESecretKey(optimizerID);
    return TFHE::GLWECipherTextType::get(context, outputKey);
  }

private:
  concrete_optimizer::dag::CircuitSolution solution;
};

} // end anonymous namespace

std::unique_ptr<mlir::OperationPass<>>
createTFHECircuitSolutionParametrizationPass(
    concrete_optimizer::dag::CircuitSolution solution) {
  return std::make_unique<TFHECircuitSolutionParametrizationPass>(solution);
}

} // namespace concretelang
} // namespace mlir
