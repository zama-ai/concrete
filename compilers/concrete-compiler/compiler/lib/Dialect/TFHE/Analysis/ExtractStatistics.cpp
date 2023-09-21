#include <concretelang/Analysis/StaticLoops.h>
#include <concretelang/Analysis/Utils.h>
#include <concretelang/Dialect/TFHE/Analysis/ExtractStatistics.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>

#include <concretelang/Dialect/TFHE/IR/TFHEOps.h>

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

struct ExtractTFHEStatisticsPass
    : public PassWrapper<ExtractTFHEStatisticsPass, OperationPass<ModuleOp>> {

  CompilationFeedback &feedback;

  ExtractTFHEStatisticsPass(CompilationFeedback &feedback)
      : feedback{feedback} {};

  void runOnOperation() override {
    WalkResult walk =
        getOperation()->walk([&](Operation *op, const WalkStage &stage) {
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
    std::optional<int64_t> numberOfIterations = tryGetStaticTripCount(op);

    if (!numberOfIterations.has_value()) {
      return StringError("only static loops can be analyzed");
    }

    assert(numberOfIterations.value() > 0);
    pass.iterations *= (uint64_t)numberOfIterations.value();
    return std::nullopt;
  }

  static std::optional<StringError> on_exit(scf::ForOp &op,
                                            ExtractTFHEStatisticsPass &pass) {
    std::optional<int64_t> numberOfIterations = tryGetStaticTripCount(op);

    if (!numberOfIterations.has_value()) {
      return StringError("only static loops can be analyzed");
    }

    assert(numberOfIterations.value() > 0);
    pass.iterations /= (uint64_t)numberOfIterations.value();
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
    auto keys = std::vector<std::pair<KeyType, size_t>>();
    auto count = pass.iterations;

    std::pair<KeyType, size_t> key =
        std::make_pair(KeyType::SECRET, (size_t)resultingKey->index);
    keys.push_back(key);

    pass.feedback.statistics.push_back(concretelang::Statistic{
        location,
        operation,
        keys,
        count,
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
    auto keys = std::vector<std::pair<KeyType, size_t>>();
    auto count = pass.iterations;

    std::pair<KeyType, size_t> key =
        std::make_pair(KeyType::SECRET, (size_t)resultingKey->index);
    keys.push_back(key);

    pass.feedback.statistics.push_back(concretelang::Statistic{
        location,
        operation,
        keys,
        count,
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
    auto keys = std::vector<std::pair<KeyType, size_t>>();
    auto count = pass.iterations;

    std::pair<KeyType, size_t> key =
        std::make_pair(KeyType::BOOTSTRAP, (size_t)bsk.getIndex());
    keys.push_back(key);

    pass.feedback.statistics.push_back(concretelang::Statistic{
        location,
        operation,
        keys,
        count,
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
    auto keys = std::vector<std::pair<KeyType, size_t>>();
    auto count = pass.iterations;

    std::pair<KeyType, size_t> key =
        std::make_pair(KeyType::KEY_SWITCH, (size_t)ksk.getIndex());
    keys.push_back(key);

    pass.feedback.statistics.push_back(concretelang::Statistic{
        location,
        operation,
        keys,
        count,
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
    auto keys = std::vector<std::pair<KeyType, size_t>>();
    auto count = pass.iterations;

    std::pair<KeyType, size_t> key =
        std::make_pair(KeyType::SECRET, (size_t)resultingKey->index);
    keys.push_back(key);

    pass.feedback.statistics.push_back(concretelang::Statistic{
        location,
        operation,
        keys,
        count,
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
    auto keys = std::vector<std::pair<KeyType, size_t>>();
    auto count = pass.iterations;

    std::pair<KeyType, size_t> key =
        std::make_pair(KeyType::SECRET, (size_t)resultingKey->index);
    keys.push_back(key);

    pass.feedback.statistics.push_back(concretelang::Statistic{
        location,
        operation,
        keys,
        count,
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
    auto keys = std::vector<std::pair<KeyType, size_t>>();
    auto count = pass.iterations;

    std::pair<KeyType, size_t> key =
        std::make_pair(KeyType::SECRET, (size_t)resultingKey->index);
    keys.push_back(key);

    // clear - encrypted = clear + neg(encrypted)

    auto operation = PrimitiveOperation::ENCRYPTED_NEGATION;
    pass.feedback.statistics.push_back(concretelang::Statistic{
        location,
        operation,
        keys,
        count,
    });

    operation = PrimitiveOperation::CLEAR_ADDITION;
    pass.feedback.statistics.push_back(concretelang::Statistic{
        location,
        operation,
        keys,
        count,
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
    auto keys = std::vector<std::pair<KeyType, size_t>>();
    auto count = pass.iterations;

    std::pair<KeyType, size_t> key =
        std::make_pair(KeyType::BOOTSTRAP, (size_t)bsk.getIndex());
    keys.push_back(key);

    key = std::make_pair(KeyType::KEY_SWITCH, (size_t)ksk.getIndex());
    keys.push_back(key);

    key = std::make_pair(KeyType::PACKING_KEY_SWITCH, (size_t)pksk.getIndex());
    keys.push_back(key);

    pass.feedback.statistics.push_back(concretelang::Statistic{
        location,
        operation,
        keys,
        count,
    });

    return std::nullopt;
  }

  size_t iterations = 1;
};

} // namespace TFHE

std::unique_ptr<OperationPass<ModuleOp>>
createStatisticExtractionPass(CompilationFeedback &feedback) {
  return std::make_unique<TFHE::ExtractTFHEStatisticsPass>(feedback);
}

} // namespace concretelang
} // namespace mlir
