#include <concretelang/Analysis/StaticLoops.h>
#include <concretelang/Analysis/Utils.h>
#include <concretelang/Dialect/TFHE/Analysis/ExtractStatistics.h>
#include <concretelang/Dialect/TFHE/IR/TFHEOps.h>
#include <concretelang/Support/CompilationFeedback.h>

#include <concrete-optimizer.hpp>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>

using namespace mlir::concretelang;
using namespace mlir;

namespace mlir {
namespace concretelang {
namespace TFHE {

#define DISPATCH_ENTER(type)                                                   \
  if (auto typedOp = llvm::dyn_cast<type>(op)) {                               \
    std::optional<StringError> error = on_enter(typedOp, *this);               \
    if (error.has_value()) {                                                   \
      return error;                                                            \
    }                                                                          \
  }

#define DISPATCH_EXIT(type)                                                    \
  if (auto typedOp = llvm::dyn_cast<type>(op)) {                               \
    std::optional<StringError> error = on_exit(typedOp, *this);                \
    if (error.has_value()) {                                                   \
      return error;                                                            \
    }                                                                          \
  }

double levelledComplexity(GLWESecretKey key, std::optional<int64_t> count) {
  double complexity = 0.0;
  if (key.isNormalized()) {
    complexity = key.getNormalized()->dimension;
  } else if (key.isParameterized()) {
    complexity = key.getParameterized()->dimension;
  }
  return complexity * count.value_or(1.0);
}

struct ExtractTFHEStatisticsPass
    : public PassWrapper<ExtractTFHEStatisticsPass, OperationPass<ModuleOp>>,
      public TripCountTracker {

  ProgramCompilationFeedback &feedback;
  CircuitCompilationFeedback *circuitFeedback;

  ExtractTFHEStatisticsPass(ProgramCompilationFeedback &feedback)
      : feedback{feedback}, circuitFeedback{nullptr} {};

  void runOnOperation() override {
    auto module = getOperation();
    auto funcs = module.getOps<mlir::func::FuncOp>();
    for (CircuitCompilationFeedback &circuitFeedback :
         feedback.circuitFeedbacks) {
      auto funcOp = llvm::find_if(funcs, [&](mlir::func::FuncOp op) {
        return op.getName() == circuitFeedback.name;
      });
      assert(funcOp != funcs.end());
      this->circuitFeedback = &circuitFeedback;

      WalkResult walk =
          (*funcOp)->walk([&](Operation *op, const WalkStage &stage) {
            if (stage.isBeforeAllRegions()) {
              std::optional<StringError> error = this->enter(op);
              if (error.has_value()) {
                op->emitError() << error->mesg;
                return WalkResult::interrupt();
              }
            }

            if (stage.isAfterAllRegions()) {
              std::optional<StringError> error = this->exit(op);
              if (error.has_value()) {
                op->emitError() << error->mesg;
                return WalkResult::interrupt();
              }
            }

            return WalkResult::advance();
          });

      if (walk.wasInterrupted()) {
        signalPassFailure();
        return;
      }
    }
  }

  std::optional<StringError> enter(mlir::Operation *op) {
    DISPATCH_ENTER(scf::ForOp)
    DISPATCH_ENTER(TFHE::AddGLWEOp)
    DISPATCH_ENTER(TFHE::AddGLWEIntOp)
    DISPATCH_ENTER(TFHE::BootstrapGLWEOp)
    DISPATCH_ENTER(TFHE::KeySwitchGLWEOp)
    DISPATCH_ENTER(TFHE::MulGLWEIntOp)
    DISPATCH_ENTER(TFHE::NegGLWEOp)
    DISPATCH_ENTER(TFHE::SubGLWEIntOp)
    DISPATCH_ENTER(TFHE::WopPBSGLWEOp)
    return std::nullopt;
  }

  std::optional<StringError> exit(mlir::Operation *op) {
    DISPATCH_EXIT(scf::ForOp)
    return std::nullopt;
  }

  static std::optional<StringError> on_enter(scf::ForOp &op,
                                             ExtractTFHEStatisticsPass &pass) {
    std::optional<int64_t> tripCount = tryGetStaticTripCount(op);

    if (!tripCount.has_value()) {
      emitWarning(op.getLoc(), "Cannot determine static trip count");
    }

    pass.pushTripCount(op, tripCount);

    return std::nullopt;
  }

  static std::optional<StringError> on_exit(scf::ForOp &op,
                                            ExtractTFHEStatisticsPass &pass) {
    std::optional<int64_t> tripCount = tryGetStaticTripCount(op);
    pass.popTripCount(op, tripCount);

    return std::nullopt;
  }

  // #############
  // TFHE.add_glwe
  // #############

  static std::optional<StringError> on_enter(TFHE::AddGLWEOp &op,
                                             ExtractTFHEStatisticsPass &pass) {
    auto resultingKey = op.getType().getKey().getNormalized();

    auto location = locationString(op.getLoc());
    auto operation = PrimitiveOperation::ENCRYPTED_ADDITION;
    auto keys = std::vector<std::pair<KeyType, int64_t>>();
    auto count = pass.getTripCount();

    std::pair<KeyType, int64_t> key =
        std::make_pair(KeyType::SECRET, (int64_t)resultingKey->index);
    keys.push_back(key);

    double complexity =
        levelledComplexity(op.getResult().getType().getKey(), count);

    pass.circuitFeedback->statistics.push_back(concretelang::Statistic{
        location,
        operation,
        keys,
        count,
        complexity,
    });

    return std::nullopt;
  }

  // #################
  // TFHE.add_glwe_int
  // #################

  static std::optional<StringError> on_enter(TFHE::AddGLWEIntOp &op,
                                             ExtractTFHEStatisticsPass &pass) {
    auto resultingKey = op.getType().getKey().getNormalized();

    auto location = locationString(op.getLoc());
    auto operation = PrimitiveOperation::CLEAR_ADDITION;
    auto keys = std::vector<std::pair<KeyType, int64_t>>();
    auto count = pass.getTripCount();

    std::pair<KeyType, int64_t> key =
        std::make_pair(KeyType::SECRET, (int64_t)resultingKey->index);
    keys.push_back(key);

    double complexity =
        levelledComplexity(op.getResult().getType().getKey(), count);

    pass.circuitFeedback->statistics.push_back(concretelang::Statistic{
        location,
        operation,
        keys,
        count,
        complexity,
    });

    return std::nullopt;
  }

  // ###################
  // TFHE.bootstrap_glwe
  // ###################

  static std::optional<StringError> on_enter(TFHE::BootstrapGLWEOp &op,
                                             ExtractTFHEStatisticsPass &pass) {
    auto bsk = op.getKey();

    auto location = locationString(op.getLoc());
    auto operation = PrimitiveOperation::PBS;
    auto keys = std::vector<std::pair<KeyType, int64_t>>();
    auto count = pass.getTripCount();

    std::pair<KeyType, int64_t> key =
        std::make_pair(KeyType::BOOTSTRAP, (int64_t)bsk.getIndex());
    keys.push_back(key);

    auto complexity =
        (double)op.getKeyAttr().getComplexity() * (double)count.value_or(1);

    pass.circuitFeedback->statistics.push_back(concretelang::Statistic{
        location,
        operation,
        keys,
        count,
        complexity,
    });

    return std::nullopt;
  }

  // ###################
  // TFHE.keyswitch_glwe
  // ###################

  static std::optional<StringError> on_enter(TFHE::KeySwitchGLWEOp &op,
                                             ExtractTFHEStatisticsPass &pass) {
    auto ksk = op.getKey();

    auto location = locationString(op.getLoc());
    auto operation = PrimitiveOperation::KEY_SWITCH;
    auto keys = std::vector<std::pair<KeyType, int64_t>>();
    auto count = pass.getTripCount();

    std::pair<KeyType, int64_t> key =
        std::make_pair(KeyType::KEY_SWITCH, (int64_t)ksk.getIndex());
    keys.push_back(key);

    auto complexity =
        (double)op.getKeyAttr().getComplexity() * (double)count.value_or(1);

    pass.circuitFeedback->statistics.push_back(concretelang::Statistic{
        location,
        operation,
        keys,
        count,
        complexity,
    });

    return std::nullopt;
  }

  // #################
  // TFHE.mul_glwe_int
  // #################

  static std::optional<StringError> on_enter(TFHE::MulGLWEIntOp &op,
                                             ExtractTFHEStatisticsPass &pass) {
    auto resultingKey = op.getType().getKey().getNormalized();

    auto location = locationString(op.getLoc());
    auto operation = PrimitiveOperation::CLEAR_MULTIPLICATION;
    auto keys = std::vector<std::pair<KeyType, int64_t>>();
    auto count = pass.getTripCount();

    std::pair<KeyType, int64_t> key =
        std::make_pair(KeyType::SECRET, (int64_t)resultingKey->index);
    keys.push_back(key);

    double complexity =
        levelledComplexity(op.getResult().getType().getKey(), count);

    pass.circuitFeedback->statistics.push_back(concretelang::Statistic{
        location,
        operation,
        keys,
        count,
        complexity,
    });

    return std::nullopt;
  }

  // #############
  // TFHE.neg_glwe
  // #############

  static std::optional<StringError> on_enter(TFHE::NegGLWEOp &op,
                                             ExtractTFHEStatisticsPass &pass) {
    auto resultingKey = op.getType().getKey().getNormalized();

    auto location = locationString(op.getLoc());
    auto operation = PrimitiveOperation::ENCRYPTED_NEGATION;
    auto keys = std::vector<std::pair<KeyType, int64_t>>();
    auto count = pass.getTripCount();

    std::pair<KeyType, int64_t> key =
        std::make_pair(KeyType::SECRET, (int64_t)resultingKey->index);
    keys.push_back(key);

    double complexity =
        levelledComplexity(op.getResult().getType().getKey(), count);

    pass.circuitFeedback->statistics.push_back(concretelang::Statistic{
        location,
        operation,
        keys,
        count,
        complexity,
    });

    return std::nullopt;
  }

  // #################
  // TFHE.sub_int_glwe
  // #################

  static std::optional<StringError> on_enter(TFHE::SubGLWEIntOp &op,
                                             ExtractTFHEStatisticsPass &pass) {
    auto resultingKey = op.getType().getKey().getNormalized();

    auto location = locationString(op.getLoc());
    auto keys = std::vector<std::pair<KeyType, int64_t>>();
    auto count = pass.getTripCount();

    std::pair<KeyType, int64_t> key =
        std::make_pair(KeyType::SECRET, (int64_t)resultingKey->index);
    keys.push_back(key);

    double complexity =
        levelledComplexity(op.getResult().getType().getKey(), count);

    // TODO: I though subtraction was implemented like this but it's complexity
    // seems to be the same as either `neg(encrypted)` or the addition, not
    // both. What should we do here?

    // clear - encrypted = clear + neg(encrypted)

    auto operation = PrimitiveOperation::ENCRYPTED_NEGATION;

    pass.circuitFeedback->statistics.push_back(concretelang::Statistic{
        location,
        operation,
        keys,
        count,
        complexity,
    });

    operation = PrimitiveOperation::CLEAR_ADDITION;

    pass.circuitFeedback->statistics.push_back(concretelang::Statistic{
        location,
        operation,
        keys,
        count,
        complexity,
    });

    return std::nullopt;
  }

  // #################
  // TFHE.wop_pbs_glwe
  // #################

  static std::optional<StringError> on_enter(TFHE::WopPBSGLWEOp &op,
                                             ExtractTFHEStatisticsPass &pass) {
    auto bsk = op.getBsk();
    auto ksk = op.getKsk();
    auto pksk = op.getPksk();

    auto location = locationString(op.getLoc());
    auto operation = PrimitiveOperation::WOP_PBS;
    auto keys = std::vector<std::pair<KeyType, int64_t>>();
    auto count = pass.getTripCount();

    std::pair<KeyType, int64_t> key =
        std::make_pair(KeyType::BOOTSTRAP, (int64_t)bsk.getIndex());
    keys.push_back(key);

    key = std::make_pair(KeyType::KEY_SWITCH, (int64_t)ksk.getIndex());
    keys.push_back(key);

    key = std::make_pair(KeyType::PACKING_KEY_SWITCH, (int64_t)pksk.getIndex());
    keys.push_back(key);

    // TODO
    double complexity = 0.0;

    pass.circuitFeedback->statistics.push_back(concretelang::Statistic{
        location,
        operation,
        keys,
        count,
        complexity,
    });

    return std::nullopt;
  }
};

} // namespace TFHE

std::unique_ptr<OperationPass<ModuleOp>>
createStatisticExtractionPass(ProgramCompilationFeedback &feedback) {
  return std::make_unique<TFHE::ExtractTFHEStatisticsPass>(feedback);
}

} // namespace concretelang
} // namespace mlir
