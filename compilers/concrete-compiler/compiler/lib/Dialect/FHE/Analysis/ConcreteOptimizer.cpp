// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include <chrono>
#include <cmath>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <vector>

#include "boost/outcome.h"

#include "concretelang/Dialect/FHE/Interfaces/FHEInterfaces.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

#include "concrete-optimizer.hpp"

#include "concretelang/Common/Error.h"
#include "concretelang/Dialect/FHE/Analysis/ConcreteOptimizer.h"
#include "concretelang/Dialect/FHE/Analysis/utils.h"
#include "concretelang/Dialect/FHE/IR/FHEAttrs.h"
#include "concretelang/Dialect/FHE/IR/FHEOps.h"
#include "concretelang/Dialect/FHE/IR/FHETypes.h"
#include "concretelang/Dialect/FHELinalg/IR/FHELinalgOps.h"
#include "concretelang/Dialect/Tracing/IR/TracingOps.h"
#include "concretelang/Support/V0Parameters.h"
#include "concretelang/Support/logging.h"

#define GEN_PASS_CLASSES
#include "concretelang/Dialect/FHE/Analysis/ConcreteOptimizer.h.inc"

using concrete_optimizer::utils::location_from_string;
using concrete_optimizer::utils::location_unknown;

namespace mlir {
namespace concretelang {
namespace optimizer {

namespace {

template <typename T> rust::Slice<const T> slice(const std::vector<T> &vec) {
  return rust::Slice<const T>(vec.data(), vec.size());
}

template <typename T> rust::Slice<const T> slice(const llvm::ArrayRef<T> &vec) {
  return rust::Slice<const T>(vec.data(), vec.size());
}

struct FunctionToDag {
  // Inputs of operators
  using Inputs = std::vector<concrete_optimizer::dag::OperatorIndex>;

  const double NEGLIGIBLE_COMPLEXITY = 0.0;

  mlir::func::FuncOp func;
  optimizer::Config config;
  llvm::DenseMap<mlir::Value, concrete_optimizer::dag::OperatorIndex> index;
  bool setOptimizerID;
  concrete_optimizer::DagBuilder &dagBuilder;

  FunctionToDag(mlir::func::FuncOp func, optimizer::Config config,
                concrete_optimizer::DagBuilder &dagBuilder)
      : func(func), config(config), dagBuilder(dagBuilder) {
    setOptimizerID = config.strategy == optimizer::Strategy::DAG_MULTI;
  }

#define DEBUG(MSG)                                                             \
  if (mlir::concretelang::isVerbose()) {                                       \
    mlir::concretelang::log_verbose() << MSG << "\n";                          \
  }

  void build() {
    // Converting arguments as Input
    mlir::Builder builder(func.getContext());
    for (size_t i = 0; i < func.getNumArguments(); i++) {
      auto arg = func.getArgument(i);
      auto optimizerIdx = addArg(arg);
      if (optimizerIdx.has_value() && setOptimizerID) {
        func.setArgAttr(i, "TFHE.OId",
                        builder.getI32IntegerAttr(optimizerIdx->index));
      }
    }
    // Converting ops
    for (auto &bb : func.getBody().getBlocks()) {
      for (auto &op : bb.getOperations()) {
        addOperation(op);
      }
    }
    for (auto &bb : func.getBody().getBlocks()) {
      for (auto &op : bb.getOperations()) {
        op.removeAttr("SMANP");
      }
    }
    if (index.empty()) {
      // Dag is empty <=> classical function without encryption
      DEBUG("!!! concrete-optimizer: nothing to do in " << func.getName()
                                                        << "\n");
      return;
    };
    DEBUG(std::string(dagBuilder.dump()));
  }

  std::optional<concrete_optimizer::dag::OperatorIndex>
  addArg(mlir::Value &arg) {
    DEBUG("Arg " << arg << " " << arg.getType());
    if (!fhe::utils::isEncryptedValue(arg)) {
      return std::nullopt;
    }
    auto precision = fhe::utils::getEintPrecision(arg);
    auto shape = getShape(arg);
    auto opI =
        dagBuilder.add_input(precision, slice(shape),
                             *loc_to_location(arg.getParentRegion()->getLoc()));
    index[arg] = opI;
    return opI;
  }

  bool hasEncryptedResult(mlir::Operation &op) {
    for (auto val : op.getResults()) {
      if (fhe::utils::isEncryptedValue(val)) {
        return true;
      }
    }
    return false;
  }

  void addOperation(mlir::Operation &op) {
    DEBUG("Instr " << op);

    auto encrypted_inputs = encryptedInputs(op);
    if (isReturn(op)) {
      for (auto op : encrypted_inputs) {
        dagBuilder.tag_operator_as_output(op);
      }
      return;
    }
    if (!hasEncryptedResult(op)) {
      // This op is unrelated to FHE
      assert(encrypted_inputs.empty() ||
             mlir::isa<mlir::concretelang::Tracing::TraceCiphertextOp>(op));
      return;
    }

    assert(op.getNumResults() == 1);
    auto val = op.getResult(0);
    auto precision = fhe::utils::getEintPrecision(val);
    concrete_optimizer::dag::OperatorIndex index;
    if (auto inputType = isLut(op); inputType != nullptr) {
      addLut(op, inputType, encrypted_inputs, precision);
      return;
    } else if (isChangePartition(op)) {
      index = addChangePartition(op, val, encrypted_inputs);
    } else if (isRound(op)) {
      index = addRound(val, encrypted_inputs, precision);
    } else if (isReinterpretPrecision(op)) {
      index = addReinterpretPrecision(val, encrypted_inputs, precision);
    } else if (auto lsb = asLsb(op)) {
      addLsb(lsb, encrypted_inputs);
      return;
    } else if (auto lsb = asLsbTensor(op)) {
      addLsb(lsb, encrypted_inputs);
      return;
    } else if (auto dot = asDot(op)) {
      auto weightsOpt = dotWeights(dot);
      if (weightsOpt) {
        index = addDot(val, encrypted_inputs, weightsOpt.value());
      } else {
        // If can't find weights return default leveled op
        DEBUG("Replace Dot by LevelledOp on " << op);
        index = addLevelledOp(op, encrypted_inputs);
      }
    } else if (auto dot = asDotEint(op)) {
      addDotEint(dot, encrypted_inputs, precision);
      // The above function call sets the OIds, can return right away
      return;
    } else if (auto mul = asMul(op)) {
      // special case as mul are rewritten in several optimizer nodes
      addMul(mul, encrypted_inputs, precision);
      return;
    } else if (auto mul = asMulTensor(op)) {
      // special case as mul are rewritten in several optimizer nodes
      addMul(mul, encrypted_inputs, precision);
      return;
    } else if (auto max = asMax(op)) {
      // special case as max are rewritten in several optimizer nodes
      addMax(max, encrypted_inputs, precision);
      return;
    } else if (auto maxpool2d = asMaxpool2d(op)) {
      // special case as max are rewritten in several optimizer nodes
      addMaxpool2d(maxpool2d, encrypted_inputs, precision);
      return;
    } else if (auto matmulEintEint = asMatmulEintEint(op)) {
      addEncMatMulTensor(matmulEintEint, encrypted_inputs, precision);
      return;
    } else if (auto zero = asZeroNoise(op)) {
      index = addZeroNoise(zero);
    } else if (auto additive = asAdditiveNoise(op)) {
      index = addAdditiveNoise(additive, encrypted_inputs);
    } else if (isMaxNoise(op)) {
      index = addMaxNoise(op, encrypted_inputs);
    } else {
      index = addLevelledOp(op, encrypted_inputs);
    }
    mlir::Builder builder(op.getContext());
    if (setOptimizerID)
      op.setAttr("TFHE.OId", builder.getI32IntegerAttr(index.index));
  }

  void addLut(mlir::Operation &op, FHE::FheIntegerInterface inputType,
              Inputs &encrypted_inputs, int precision) {
    auto val = op.getResult(0);
    auto loc = loc_to_location(op.getLoc());
    assert(encrypted_inputs.size() == 1);
    // No need to distinguish different lut kind until we do approximate
    // paradigm on outputs
    auto encrypted_input = encrypted_inputs[0];
    std::vector<std::uint64_t> unknowFunction;
    std::vector<int32_t> operatorIndexes;
    if (inputType.isSigned()) {
      // std::vector<std::int64_t> weights_vector{1};
      auto addIndex =
          dagBuilder.add_dot(slice(encrypted_inputs),
                             concrete_optimizer::weights::number(1), *loc);
      encrypted_input = addIndex;
      operatorIndexes.push_back(addIndex.index);
    }
    auto lutIndex = dagBuilder.add_lut(encrypted_input, slice(unknowFunction),
                                       precision, *loc);
    operatorIndexes.push_back(lutIndex.index);
    mlir::Builder builder(op.getContext());
    if (setOptimizerID)
      op.setAttr("TFHE.OId", builder.getDenseI32ArrayAttr(operatorIndexes));
    index[val] = lutIndex;
  }

  concrete_optimizer::dag::OperatorIndex
  addRound(mlir::Value &val, Inputs &encrypted_inputs, int rounded_precision) {
    assert(encrypted_inputs.size() == 1);
    // No need to distinguish different lut kind until we do approximate
    // paradigm on outputs
    auto encrypted_input = encrypted_inputs[0];
    index[val] = dagBuilder.add_round_op(encrypted_input, rounded_precision,
                                         *loc_to_location(val.getLoc()));
    return index[val];
  }

  concrete_optimizer::dag::OperatorIndex
  addChangePartition(mlir::Operation &op, mlir::Value &val,
                     Inputs &encrypted_inputs) {
    assert(encrypted_inputs.size() == 1);
    auto encrypted_input = encrypted_inputs[0];
    auto options = options_from_config(config);

    auto partitionBuilder =
        [&options](mlir::concretelang::FHE::PartitionAttr partitionAttr) {
          auto glweDim = partitionAttr.getGlweDim();
          auto lweDim = partitionAttr.getLweDim();
          auto pbsLevel = partitionAttr.getPbsLevel();
          auto pbsLogBase = partitionAttr.getPbsBaseLog();
          auto doubleLogPolySize = log2(partitionAttr.getPolySize());
          assert(ceil(doubleLogPolySize) == doubleLogPolySize &&
                 "polySize is not a power of 2");
          auto logPolySize = (uint64_t)doubleLogPolySize;
          auto max_variance = concrete_optimizer::utils::get_noise_br(
              options, logPolySize, glweDim, lweDim, pbsLevel, pbsLogBase);
          auto name = partitionAttr.getName().getValue().str();
          // TODO: max_variance vs variance
          return concrete_optimizer::utils::get_external_partition(
              name, logPolySize, glweDim, lweDim, max_variance, max_variance);
        };

    if (auto srcPartitionAttr =
            op.getAttrOfType<mlir::concretelang::FHE::PartitionAttr>("src")) {
      assert(
          !op.getAttrOfType<mlir::concretelang::FHE::PartitionAttr>("dest") &&
          "ChangePartition: can't have both src and dest partitions set");

      auto partition = partitionBuilder(srcPartitionAttr);

      index[val] = dagBuilder.add_change_partition_with_src(
          encrypted_input, *partition, *loc_to_location(val.getLoc()));
    } else if (auto destPartitionAttr =
                   op.getAttrOfType<mlir::concretelang::FHE::PartitionAttr>(
                       "dest")) {
      auto partition = partitionBuilder(destPartitionAttr);

      index[val] = dagBuilder.add_change_partition_with_dst(
          encrypted_input, *partition, *loc_to_location(val.getLoc()));
    } else {
      assert(false &&
             "ChangePartition: one of src or dest partitions need to be set");
    }
    return index[val];
  }

  concrete_optimizer::dag::OperatorIndex
  addReinterpretPrecision(mlir::Value &val, Inputs &encrypted_inputs,
                          int new_precision) {
    assert(encrypted_inputs.size() == 1);
    auto encrypted_input = encrypted_inputs[0];
    index[val] = dagBuilder.add_unsafe_cast_op(encrypted_input, new_precision,
                                               *loc_to_location(val.getLoc()));
    return index[val];
  }

  concrete_optimizer::dag::OperatorIndex
  addDot(mlir::Value &val, Inputs &encrypted_inputs,
         std::vector<std::int64_t> &weights_vector) {
    assert(encrypted_inputs.size() == 1);
    auto weights = concrete_optimizer::weights::vector(slice(weights_vector));
    index[val] = dagBuilder.add_dot(slice(encrypted_inputs), std::move(weights),
                                    *loc_to_location(val.getLoc()));
    return index[val];
  }

  std::string loc_to_string(mlir::Location location) {
    std::string loc;
    llvm::raw_string_ostream loc_stream(loc);
    location.print(loc_stream);
    loc.erase(0, 4); // erase the loc( prefix
    loc.pop_back();  // erase the ) postfix
    loc.erase(std::remove(loc.begin(), loc.end(), '"'),
              loc.end()); // erase the " characters
    return loc;
  }

  concrete_optimizer::dag::OperatorIndex
  addZeroNoise(concretelang::FHE::ZeroNoise &op) {
    auto val = op->getOpResult(0);
    auto outShape = getShape(val);
    auto loc = loc_to_location(op.getLoc());
    auto precision = fhe::utils::getEintPrecision(val);
    index[val] = dagBuilder.add_zero_noise(precision, slice(outShape), *loc);
    return index[val];
  }

  concrete_optimizer::dag::OperatorIndex
  addAdditiveNoise(concretelang::FHE::AdditiveNoise &op, Inputs &inputs) {
    auto val = op->getResult(0);
    auto out_shape = getShape(val);
    // Default complexity is negligible
    double fixed_cost = NEGLIGIBLE_COMPLEXITY;
    double lwe_dim_cost_factor = NEGLIGIBLE_COMPLEXITY;
    auto comment = std::string(op->getName().getStringRef()) + " " +
                   loc_to_string(op.getLoc());
    auto loc = loc_to_location(op.getLoc());
    auto weights = std::vector<double>(inputs.size(), 1.);
    index[val] = dagBuilder.add_linear_noise(slice(inputs), lwe_dim_cost_factor,
                                             fixed_cost, slice(weights),
                                             slice(out_shape), comment, *loc);
    return index[val];
  }

  rust::Box<concrete_optimizer::Location>
  loc_to_location(mlir::Location location) {
    return location_from_string(loc_to_string(location));
  }

  concrete_optimizer::dag::OperatorIndex addMaxNoise(mlir::Operation &op,
                                                     Inputs &inputs) {
    auto val = op.getResult(0);
    auto out_shape = getShape(val);
    auto loc = loc_to_location(op.getLoc());
    assert(!inputs.empty());

    index[val] =
        dagBuilder.add_max_noise(slice(inputs), slice(out_shape), *loc);
    return index[val];
  }

  concrete_optimizer::dag::OperatorIndex addLevelledOp(mlir::Operation &op,
                                                       Inputs &inputs) {
    auto val = op.getResult(0);
    auto out_shape = getShape(val);
    auto loc = loc_to_location(op.getLoc());
    if (inputs.empty()) {
      // Trivial encrypted constants encoding
      // There are converted to input + levelledop
      auto precision = fhe::utils::getEintPrecision(val);
      auto opI = dagBuilder.add_input(precision, slice(out_shape), *loc);
      inputs.push_back(opI);
    }
    // Default complexity is negligible
    double fixed_cost = NEGLIGIBLE_COMPLEXITY;
    double lwe_dim_cost_factor = NEGLIGIBLE_COMPLEXITY;
    auto smanp_int = op.getAttrOfType<mlir::IntegerAttr>("SMANP");
    assert(smanp_int && "Missing manp value on a crypto operation");
    auto comment = std::string(op.getName().getStringRef()) + " " +
                   loc_to_string(op.getLoc());

    size_t n_inputs = 0;
    double sq_sum = 0;
    for (auto input : op.getOperands()) {
      if (!fhe::utils::isEncryptedValue(input)) {
        continue;
      }
      n_inputs += 1;
      if (input.isa<mlir::BlockArgument>()) {
        sq_sum += 1.0;
      } else {
        auto inpSmanpInt =
            input.getDefiningOp()->getAttrOfType<mlir::IntegerAttr>("SMANP");
        const double inpSManp = inpSmanpInt.getValue().roundToDouble();
        sq_sum += inpSManp;
      }
    }
    assert(inputs.size() == n_inputs);
    double weight;
    if (sq_sum == 0) {
      // The max input manp is zero, meaning the inputs are all zero tensors
      // with no noise. In this case it does not matter the weight since it will
      // multiply zero.
      weight = 1.;
    } else {
      double smanp_dbl = smanp_int.getValue().roundToDouble();
      weight = std::max(sqrt(smanp_dbl / sq_sum), 1.0);
      assert(!std::isnan(weight));
    }
    auto weights = std::vector<double>(n_inputs, weight);
    index[val] = dagBuilder.add_linear_noise(slice(inputs), lwe_dim_cost_factor,
                                             fixed_cost, slice(weights),
                                             slice(out_shape), comment, *loc);
    return index[val];
  }

  bool isSignedEint(mlir::Type type) {
    if (auto tensor = type.dyn_cast<RankedTensorType>(); tensor != nullptr) {
      type = tensor.getElementType();
    }
    return type.cast<FHE::FheIntegerInterface>().isSigned();
  }

  template <typename LsbOp>
  void addLsb(LsbOp &lsbOp, Inputs &encrypted_inputs) {
    assert(encrypted_inputs.size() == 1);
    auto input = lsbOp.getInput();
    auto result = lsbOp.getResult();
    auto loc = loc_to_location(lsbOp.getLoc());
    auto input_precision = fhe::utils::getEintPrecision(input);
    auto output_precision = fhe::utils::getEintPrecision(result);
    auto lsb_shiffted_as_1bit_wop = dagBuilder.add_dot(
        slice(encrypted_inputs),
        concrete_optimizer::weights::number(1 << input_precision), *loc);
    std::vector<std::uint64_t> unknownFunction;
    auto overflow_bit_precision = 0;
    auto lsb_as_0_bits = dagBuilder.add_unsafe_cast_op(lsb_shiffted_as_1bit_wop,
                                                       overflow_bit_precision,
                                                       *loc); // id for rotation
    auto lsb_result = dagBuilder.add_lut(lsb_as_0_bits, slice(unknownFunction),
                                         output_precision, *loc);
    auto lsb_result_corrected = idPlaceholder(lsb_result);
    index[result] = lsb_result_corrected;

    if (!setOptimizerID) {
      return;
    }

    mlir::SmallVector<int32_t, 3> operatorIndexes = {
        // see `extractBitWithClearedLowerBits` in
        // lib/Conversion/FHEToTFHEScalar/FHEToTFHEScalar.cpp
        (int32_t)lsb_shiffted_as_1bit_wop.index, // for shift
        (int32_t)lsb_as_0_bits.index,            // for rotation
        (int32_t)lsb_result.index,               // for ks
        (int32_t)lsb_result.index,               // for bootstrap
        (int32_t)lsb_result_corrected.index,     // for correction
    };
    mlir::Builder builder(lsbOp.getContext());
    lsbOp->setAttr("TFHE.OId", builder.getDenseI32ArrayAttr(operatorIndexes));
  }

  template <typename MulOp>
  void addMul(MulOp &mulOp, Inputs &inputs, int precision) {

    // x * y = ((x + y)^2 / 4) - ((x - y)^2 / 4) == tlu(x + y) - tlu(x - y)

    mlir::Value result = mulOp.getResult();
    const std::vector<uint64_t> resultShape = getShape(result);

    const double fixedCost = NEGLIGIBLE_COMPLEXITY;
    const double lweDimCostFactor = NEGLIGIBLE_COMPLEXITY;

    auto comment = std::string(mulOp->getName().getStringRef()) + " " +
                   loc_to_string(mulOp.getLoc());
    auto loc = loc_to_location(mulOp.getLoc());

    // for tlus
    const std::vector<std::uint64_t> unknownFunction;

    // tlu(x + y)

    auto addWeights = std::vector<double>{1, 1};
    auto addNode = dagBuilder.add_linear_noise(
        slice(inputs), lweDimCostFactor, fixedCost, slice(addWeights),
        slice(resultShape), comment, *loc);

    std::optional<concrete_optimizer::dag::OperatorIndex> lhsCorrectionNode;
    if (isSignedEint(mulOp.getType())) {
      // If signed mul we need to add the addition node for correction of the
      // signed tlu
      addNode = dagBuilder.add_dot(
          slice(std::vector<concrete_optimizer::dag::OperatorIndex>{addNode}),
          concrete_optimizer::weights::vector(
              slice(std::vector<std::int64_t>{1})),
          *loc);
      lhsCorrectionNode = addNode;
    }
    auto lhsTluNode =
        dagBuilder.add_lut(addNode, slice(unknownFunction), precision, *loc);

    // tlu(x - y)
    auto subWeights = std::vector<double>{1, 1};
    auto subNode = dagBuilder.add_linear_noise(
        slice(inputs), lweDimCostFactor, fixedCost, slice(subWeights),
        slice(resultShape), comment, *loc);

    // This is a signed tlu so we need to also add the addition for correction
    // signed tlu
    auto rhsCorrectionNode = dagBuilder.add_dot(
        slice(std::vector<concrete_optimizer::dag::OperatorIndex>{subNode}),
        concrete_optimizer::weights::vector(
            slice(std::vector<std::int64_t>{1})),
        *loc);
    auto rhsTluNode = dagBuilder.add_lut(
        rhsCorrectionNode, slice(unknownFunction), precision, *loc);

    // tlu(x + y) - tlu(x - y)
    auto resultWeights = std::vector<double>{1, 1};
    const std::vector<concrete_optimizer::dag::OperatorIndex> subInputs = {
        lhsTluNode, rhsTluNode};
    auto resultNode = dagBuilder.add_linear_noise(
        slice(subInputs), lweDimCostFactor, fixedCost, slice(resultWeights),
        slice(resultShape), comment, *loc);

    index[result] = resultNode;

    mlir::Builder builder(mulOp.getContext());
    mlir::SmallVector<int32_t, 7> operatorIndexes = {
        (int32_t)addNode.index,    (int32_t)lhsTluNode.index,
        (int32_t)subNode.index,    (int32_t)rhsCorrectionNode.index,
        (int32_t)rhsTluNode.index, (int32_t)resultNode.index,
    };
    if (lhsCorrectionNode.has_value()) {
      // We push that at the end by convention
      operatorIndexes.push_back(lhsCorrectionNode.value().index);
    }

    if (setOptimizerID)
      mulOp->setAttr("TFHE.OId", builder.getDenseI32ArrayAttr(operatorIndexes));
  }

  template <typename InnerProductOp>
  concrete_optimizer::dag::OperatorIndex
  addTensorInnerProductEncEnc(InnerProductOp &innerProductOp, Inputs &inputs,
                              int precision) {
    mlir::Value result = innerProductOp.getResult();
    auto loc = loc_to_location(innerProductOp.getLoc());
    const std::vector<uint64_t> resultShape = getShape(result);

    // We assume a first tensorized matmul step
    // is the construction of matrices of:
    //   - sums of all pairs
    //   - differences of all pairs
    // where pairs are pairs of values that are to be multiplied together

    // Compute the number of elements in each of the
    // matrices of pairs

    auto lhsType = ((mlir::Type)innerProductOp.getLhs().getType())
                       .cast<mlir::RankedTensorType>();

    auto rhsType = ((mlir::Type)innerProductOp.getRhs().getType())
                       .cast<mlir::RankedTensorType>();

    std::vector<int64_t> lhsShape = lhsType.getShape();
    std::vector<int64_t> rhsShape = rhsType.getShape();

    if (rhsShape.size() == 1)
      rhsShape.push_back(1);

    if (lhsShape.size() == 1)
      lhsShape.emplace(lhsShape.begin(), 1);

    int64_t rhsDims = (int64_t)rhsShape.size();
    int64_t lhsDims = (int64_t)lhsShape.size();

    // Suppose lhsDims is (5, 3, 2) -> 5 matrices of size 3x2 (2 is the
    // reduction dimension) and     rhsDims is (3, 5, 2, 3) -> 3x 5 matrices of
    // size 2x3 the pair matrix would have size (3, 5, 3, 3, 2) this is the
    // shape of the matrix onto which we apply the TLUs that compute the
    // multiplication of all pairs of values

    // The RHS can be a (N,) matrix, the outer dimension is supposed to be 1
    int64_t rhsOuterDim = rhsShape[rhsDims - 1];
    int64_t lhsOuterDim = lhsShape[lhsDims - 2];

    std::vector<uint64_t> pairMatrixShape;

    // Compute the output matrix dimension
    // Corresponding dimensions that are considered "compatible" are "N, 1", "1,
    // N", "N, N"
    int64_t rhsDimIter = rhsDims - 3, lhsDimIter = lhsDims - 3;
    if (rhsDimIter >= 0 && lhsDimIter >= 0) {
      while (rhsDimIter >= 0 && lhsDimIter >= 0 &&
             (lhsShape[lhsDimIter] == rhsShape[rhsDimIter] ||
              lhsShape[lhsDimIter] == 1 || rhsShape[rhsDimIter] == 1)) {
        pairMatrixShape.push_back(
            std::max(rhsShape[rhsDimIter], lhsShape[lhsDimIter]));
        --lhsDimIter;
        --rhsDimIter;
      }
    }
    assert((lhsDimIter < 0 || rhsDimIter < 0) &&
           "Bad dimensions given to matmul or dot");
    while (lhsDimIter >= 0) {
      pairMatrixShape.push_back(lhsShape[lhsDimIter]);
      --lhsDimIter;
    }
    while (rhsDimIter >= 0) {
      pairMatrixShape.push_back(rhsShape[rhsDimIter]);
      --rhsDimIter;
    }
    // Add the outer dimensions of the individual matrices
    pairMatrixShape.push_back(lhsOuterDim);
    pairMatrixShape.push_back(rhsOuterDim);

    // Add the reduction dimension
    // The number of elements in the dot product
    // is the number of cells on the reduction axis (aka "destroyed dimension")
    int64_t reductionDimSize = rhsShape[rhsDims - 2];
    assert(lhsShape[lhsDims - 1] == reductionDimSize);

    pairMatrixShape.push_back(reductionDimSize);

    // Compute the manp of the various steps
    // in the matmul of enc x enc:

    // 1. (x + y) and (x - y) -> supposing broadcasting is used
    // to tensorize this operation
    const double fixedCost = NEGLIGIBLE_COMPLEXITY;
    const double lweDimCostFactor = NEGLIGIBLE_COMPLEXITY;

    auto comment = std::string(innerProductOp->getName().getStringRef()) + " " +
                   loc_to_string(innerProductOp.getLoc());

    // tlu(v)
    const double tluManp = 1;

    // tlu(v1) - tlu(v2)
    const double tluSubManp = sqrt(tluManp + tluManp);

    // for tlus
    const std::vector<std::uint64_t> unknownFunction;

    // tlu(x + y)
    auto addWeights = std::vector<double>{1, 1};
    auto addNode = dagBuilder.add_linear_noise(
        slice(inputs), lweDimCostFactor, fixedCost, slice(addWeights),
        slice(pairMatrixShape), comment, *loc);

    std::optional<concrete_optimizer::dag::OperatorIndex> lhsCorrectionNode;
    if (isSignedEint(innerProductOp.getType())) {
      // If signed mul we need to add the addition node for correction of the
      // signed tlu
      addNode = dagBuilder.add_dot(
          slice(std::vector<concrete_optimizer::dag::OperatorIndex>{addNode}),
          concrete_optimizer::weights::vector(
              slice(std::vector<std::int64_t>{1})),
          *loc);
      lhsCorrectionNode = addNode;
    }
    auto lhsTluNode =
        dagBuilder.add_lut(addNode, slice(unknownFunction), precision, *loc);

    // tlu(x - y)
    auto subWeights = std::vector<double>{1, 1};
    auto subNode = dagBuilder.add_linear_noise(
        slice(inputs), lweDimCostFactor, fixedCost, slice(subWeights),
        slice(pairMatrixShape), comment, *loc);

    // This is a signed tlu so we need to also add the addition for correction
    // signed tlu
    auto rhsCorrectionNode = dagBuilder.add_dot(
        slice(std::vector<concrete_optimizer::dag::OperatorIndex>{subNode}),
        concrete_optimizer::weights::vector(
            slice(std::vector<std::int64_t>{1})),
        *loc);
    auto rhsTluNode = dagBuilder.add_lut(
        rhsCorrectionNode, slice(unknownFunction), precision, *loc);

    // tlu(x + y) - tlu(x - y)
    auto resultWeights = std::vector<double>{1, 1};
    const std::vector<concrete_optimizer::dag::OperatorIndex> subInputs = {
        lhsTluNode, rhsTluNode};
    auto resultNode = dagBuilder.add_linear_noise(
        slice(subInputs), lweDimCostFactor, fixedCost, slice(resultWeights),
        slice(pairMatrixShape), comment, *loc);

    // 3. Sum(tlu(x + y) - tlu(x - y))
    // Create a leveled op that simulates concatenation. It takes
    // as inputs all the intermediary dot product results and produces
    // the output tensor

    // Default complexity is negligible
    double fixed_cost = NEGLIGIBLE_COMPLEXITY;
    double lwe_dim_cost_factor = NEGLIGIBLE_COMPLEXITY;

    // For the output of the operation, take the MANP from the MANP pass
    mlir::Operation *op = innerProductOp.getOperation();
    mlir::IntegerAttr smanp_int = op->getAttrOfType<mlir::IntegerAttr>("SMANP");
    assert(smanp_int && "Missing manp value on a crypto operation");

    const std::vector<concrete_optimizer::dag::OperatorIndex> sumOperands = {
        resultNode};

    // TODO: use APIFloat.sqrt when it's available
    double manp = sqrt(smanp_int.getValue().roundToDouble());
    auto weights = std::vector<double>(sumOperands.size(), manp / tluSubManp);
    index[result] = dagBuilder.add_linear_noise(
        slice(sumOperands), lwe_dim_cost_factor, fixed_cost, slice(weights),
        slice(resultShape), comment, *loc);

    // Create the TFHE.OId attributes
    // The first elements of the vector are nodes for the encrypted
    // multiplication
    mlir::Builder builder(innerProductOp.getContext());
    mlir::SmallVector<int32_t, 8> operatorIndexes = {
        (int32_t)addNode.index,    (int32_t)lhsTluNode.index,
        (int32_t)subNode.index,    (int32_t)rhsCorrectionNode.index,
        (int32_t)rhsTluNode.index, (int32_t)resultNode.index,
    };
    if (lhsCorrectionNode.has_value()) {
      operatorIndexes.push_back(lhsCorrectionNode.value().index);
    }
    // The last element of the vector is the node for the addition
    operatorIndexes.push_back((int32_t)index[result].index);

    if (setOptimizerID)
      innerProductOp->setAttr("TFHE.OId",
                              builder.getDenseI32ArrayAttr(operatorIndexes));

    return index[result];
  }

  concrete_optimizer::dag::OperatorIndex
  addEncMatMulTensor(FHELinalg::MatMulEintEintOp &matmulOp, Inputs &inputs,
                     int precision) {
    return addTensorInnerProductEncEnc<FHELinalg::MatMulEintEintOp>(
        matmulOp, inputs, precision);
  }

  concrete_optimizer::dag::OperatorIndex
  addDotEint(FHELinalg::DotEint &dotOp, Inputs &inputs, int precision) {
    return addTensorInnerProductEncEnc<FHELinalg::DotEint>(dotOp, inputs,
                                                           precision);
  }

  void addMax(FHE::MaxEintOp &maxOp, Inputs &inputs, int precision) {
    mlir::Value result = maxOp.getResult();
    auto loc = loc_to_location(maxOp.getLoc());
    const std::vector<uint64_t> resultShape = getShape(result);

    const double fixedCost = NEGLIGIBLE_COMPLEXITY;
    const double lweDimCostFactor = NEGLIGIBLE_COMPLEXITY;

    auto comment = std::string(maxOp->getName().getStringRef()) + " " +
                   loc_to_string(maxOp.getLoc());

    auto subWeights = std::vector<double>{1, 1};
    auto subNode = dagBuilder.add_linear_noise(
        slice(inputs), lweDimCostFactor, fixedCost, slice(subWeights),
        slice(resultShape), comment, *loc);

    const std::vector<std::uint64_t> unknownFunction;
    auto tluNode =
        dagBuilder.add_lut(subNode, slice(unknownFunction), precision, *loc);

    const std::vector<concrete_optimizer::dag::OperatorIndex> addInputs = {
        tluNode, inputs[1]};
    auto addWeights = std::vector<double>{1, 1};
    auto resultNode = dagBuilder.add_linear_noise(
        slice(addInputs), lweDimCostFactor, fixedCost, slice(addWeights),
        slice(resultShape), comment, *loc);

    index[result] = resultNode;

    // Set attribute on the MLIR node
    mlir::Builder builder(maxOp.getContext());
    mlir::SmallVector<int32_t, 3> operatorIndexes = {(int32_t)subNode.index,
                                                     (int32_t)tluNode.index,
                                                     (int32_t)resultNode.index};
    if (setOptimizerID)
      maxOp->setAttr("TFHE.OId", builder.getDenseI32ArrayAttr(operatorIndexes));
  }

  void addMaxpool2d(FHELinalg::Maxpool2dOp &maxpool2dOp, Inputs &inputs,
                    int precision) {
    mlir::Value result = maxpool2dOp.getResult();
    const std::vector<uint64_t> resultShape = getShape(result);
    auto loc = loc_to_location(maxpool2dOp.getLoc());

    // all TLUs are flattened into a dimension
    // to create a single TLU node in optimizer dag
    std::vector<uint64_t> fakeShape = resultShape;

    uint64_t numberOfComparisons = 1;
    for (auto dimensionSize :
         maxpool2dOp.getKernelShape().getValues<int64_t>()) {
      numberOfComparisons *= dimensionSize;
    }
    fakeShape.push_back(numberOfComparisons);

    Operation *inputOp = maxpool2dOp.getInput().getDefiningOp();

    const double fixedCost = NEGLIGIBLE_COMPLEXITY;
    const double lweDimCostFactor = NEGLIGIBLE_COMPLEXITY;

    llvm::APInt inputSmanp = llvm::APInt{1, 1, false};
    if (inputOp != nullptr) {
      const auto inputSmanpAttr =
          inputOp->getAttrOfType<mlir::IntegerAttr>("SMANP");
      assert(inputSmanpAttr && "Missing SMANP value on a crypto operation");
      inputSmanp = inputSmanpAttr.getValue();
    }

    const double subManp = sqrt(2 * inputSmanp.roundToDouble() + 1);

    auto comment = std::string(maxpool2dOp->getName().getStringRef()) + " " +
                   loc_to_string(maxpool2dOp.getLoc());

    auto subWeights = std::vector<double>(
        inputs.size(), subManp / sqrt(inputSmanp.roundToDouble()));
    auto subNode = dagBuilder.add_linear_noise(slice(inputs), lweDimCostFactor,
                                               fixedCost, slice(subWeights),
                                               slice(fakeShape), comment, *loc);

    const std::vector<std::uint64_t> unknownFunction;
    auto tluNode =
        dagBuilder.add_lut(subNode, slice(unknownFunction), precision, *loc);

    const double addManp = sqrt(inputSmanp.roundToDouble() + 1);
    const std::vector<concrete_optimizer::dag::OperatorIndex> addInputs = {
        tluNode, inputs[0]};

    auto resultWeights = std::vector<double>(
        addInputs.size(), addManp / sqrt(inputSmanp.roundToDouble()));
    auto resultNode = dagBuilder.add_linear_noise(
        slice(addInputs), lweDimCostFactor, fixedCost, slice(resultWeights),
        slice(resultShape), comment, *loc);

    index[result] = resultNode;
    // Set attribute on the MLIR node
    mlir::Builder builder(maxpool2dOp.getContext());
    mlir::SmallVector<int32_t, 3> operatorIndexes = {(int32_t)subNode.index,
                                                     (int32_t)tluNode.index,
                                                     (int32_t)resultNode.index};
    // TODO : The subtraction of the signed case is not given to the optimizer
    // which could lead to some issue with the dag partitioning of the
    // optimizer.
    // Note: Should not be an issue while the partition are computed
    // on the precision.
    if (setOptimizerID)
      maxpool2dOp->setAttr("TFHE.OId",
                           builder.getDenseI32ArrayAttr(operatorIndexes));
  }

  concrete_optimizer::dag::OperatorIndex
  idPlaceholder(concrete_optimizer::dag::OperatorIndex input) {
    std::vector inputs = {input};
    return dagBuilder.add_dot(slice(inputs),
                              concrete_optimizer::weights::number(1),
                              *location_unknown());
  }

  Inputs encryptedInputs(mlir::Operation &op) {
    Inputs inputs;
    for (auto operand : op.getOperands()) {
      auto entry = index.find(operand);
      if (entry == index.end()) {
        assert(!fhe::utils::isEncryptedValue(operand));
        DEBUG("Ignoring as input " << operand);
        continue;
      }
      inputs.push_back(entry->getSecond());
    }
    return inputs;
  }

  template <typename LinalgApplyLookupTable>
  FHE::FheIntegerInterface getEintTypeOfLut(LinalgApplyLookupTable op) {
    auto tensorType = op.getT().getType().template dyn_cast<mlir::TensorType>();
    auto eint = tensorType.getElementType()
                    .template dyn_cast<FHE::FheIntegerInterface>();
    assert(eint != nullptr);
    return eint;
  }

  // Returns the FHE integer type on which the lut is performed else return a
  // nullptr
  FHE::FheIntegerInterface isLut(mlir::Operation &op) {
    if (auto lut =
            llvm::dyn_cast<mlir::concretelang::FHE::ApplyLookupTableEintOp>(op);
        lut != nullptr) {
      auto eint = lut.getA().getType().dyn_cast<FHE::FheIntegerInterface>();
      assert(eint != nullptr);
      return eint;
    }
    if (auto lut = llvm::dyn_cast<
            mlir::concretelang::FHELinalg::ApplyLookupTableEintOp>(op);
        lut != nullptr) {
      return getEintTypeOfLut(lut);
    }
    if (auto lut = llvm::dyn_cast<
            mlir::concretelang::FHELinalg::ApplyMultiLookupTableEintOp>(op);
        lut != nullptr) {
      return getEintTypeOfLut(lut);
    }
    if (auto lut = llvm::dyn_cast<
            mlir::concretelang::FHELinalg::ApplyMappedLookupTableEintOp>(op);
        lut != nullptr) {
      return getEintTypeOfLut(lut);
    }
    return nullptr;
  }

  bool isChangePartition(mlir::Operation &op) {
    return llvm::isa<mlir::concretelang::FHE::ChangePartitionEintOp>(op);
  }

  bool isRound(mlir::Operation &op) {
    return llvm::isa<mlir::concretelang::FHE::RoundEintOp>(op) ||
           llvm::isa<mlir::concretelang::FHELinalg::RoundOp>(op);
  }

  bool isReinterpretPrecision(mlir::Operation &op) {
    return llvm::isa<mlir::concretelang::FHE::ReinterpretPrecisionEintOp>(op) ||
           llvm::isa<mlir::concretelang::FHELinalg::ReinterpretPrecisionEintOp>(
               op);
  }

  mlir::concretelang::FHELinalg::Dot asDot(mlir::Operation &op) {
    return llvm::dyn_cast<mlir::concretelang::FHELinalg::Dot>(op);
  }

  mlir::concretelang::FHELinalg::DotEint asDotEint(mlir::Operation &op) {
    return llvm::dyn_cast<mlir::concretelang::FHELinalg::DotEint>(op);
  }

  mlir::concretelang::FHE::MulEintOp asMul(mlir::Operation &op) {
    return llvm::dyn_cast<mlir::concretelang::FHE::MulEintOp>(op);
  }

  mlir::concretelang::FHELinalg::MulEintOp asMulTensor(mlir::Operation &op) {
    return llvm::dyn_cast<mlir::concretelang::FHELinalg::MulEintOp>(op);
  }

  mlir::concretelang::FHE::ZeroNoise asZeroNoise(mlir::Operation &op) {
    return llvm::dyn_cast<mlir::concretelang::FHE::ZeroNoise>(op);
  }

  mlir::concretelang::FHE::AdditiveNoise asAdditiveNoise(mlir::Operation &op) {
    return llvm::dyn_cast<mlir::concretelang::FHE::AdditiveNoise>(op);
  }

  mlir::concretelang::FHE::MaxEintOp asMax(mlir::Operation &op) {
    return llvm::dyn_cast<mlir::concretelang::FHE::MaxEintOp>(op);
  }

  mlir::concretelang::FHELinalg::Maxpool2dOp asMaxpool2d(mlir::Operation &op) {
    return llvm::dyn_cast<mlir::concretelang::FHELinalg::Maxpool2dOp>(op);
  }

  mlir::concretelang::FHELinalg::MatMulEintEintOp
  asMatmulEintEint(mlir::Operation &op) {
    return llvm::dyn_cast<mlir::concretelang::FHELinalg::MatMulEintEintOp>(op);
  }

  mlir::concretelang::FHE::LsbEintOp asLsb(mlir::Operation &op) {
    return llvm::dyn_cast<mlir::concretelang::FHE::LsbEintOp>(op);
  }

  mlir::concretelang::FHELinalg::LsbEintOp asLsbTensor(mlir::Operation &op) {
    return llvm::dyn_cast<mlir::concretelang::FHELinalg::LsbEintOp>(op);
  }

  bool isReturn(mlir::Operation &op) {
    return llvm::isa<mlir::func::ReturnOp>(op);
  }

  bool isConst(mlir::Operation &op) {
    return llvm::isa<mlir::arith::ConstantOp>(op);
  }

  bool isArg(const mlir::Value &value) {
    return value.isa<mlir::BlockArgument>();
  }

  bool isMaxNoise(mlir::Operation &op) {
    return llvm::isa<mlir::concretelang::FHE::MaxNoise>(op);
  }

  std::optional<std::vector<std::int64_t>>
  resolveConstantVectorWeights(mlir::arith::ConstantOp &cstOp) {
    std::vector<std::int64_t> values;
    mlir::DenseIntElementsAttr denseVals =
        cstOp->getAttrOfType<mlir::DenseIntElementsAttr>("value");

    for (llvm::APInt val : denseVals.getValues<llvm::APInt>()) {
      if (val.getActiveBits() > 64) {
        return std::nullopt;
      }
      values.push_back(val.getSExtValue());
    }
    return values;
  }

  std::optional<std::vector<std::int64_t>>
  resolveConstantWeights(mlir::Value &value) {
    if (auto cstOp = llvm::dyn_cast_or_null<mlir::arith::ConstantOp>(
            value.getDefiningOp())) {
      auto shape = getShape(value);
      switch (shape.size()) {
      case 1:
        return resolveConstantVectorWeights(cstOp);
      default:
        DEBUG("High-Rank tensor: rely on MANP and levelledOp");
        return std::nullopt;
      }
    } else {
      DEBUG("Dynamic Weights: rely on MANP and levelledOp");
      return std::nullopt;
    }
  }

  std::optional<std::vector<std::int64_t>>
  dotWeights(mlir::concretelang::FHELinalg::Dot &dot) {
    if (dot.getOperands().size() != 2) {
      return std::nullopt;
    }
    auto weights = dot.getOperands()[1];
    return resolveConstantWeights(weights);
  }

  std::vector<std::uint64_t> getShape(mlir::Value &value) {
    return getShape(value.getType());
  }

  std::vector<std::uint64_t> getShape(mlir::Type type_) {
    if (auto ranked_tensor = type_.dyn_cast_or_null<mlir::RankedTensorType>()) {
      std::vector<std::uint64_t> shape;
      for (auto v : ranked_tensor.getShape()) {
        shape.push_back(v);
      }
      return shape;
    } else {
      return {};
    }
  }
};

} // namespace

struct DagPass : ConcreteOptimizerBase<DagPass> {
  optimizer::Config config;
  concrete_optimizer::Dag &dag;

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    auto name = std::string(func.getName());
    DEBUG("ConcreteOptimizer Dag: " << name);
    auto builder = dag.builder(name);
    FunctionToDag(func, config, *builder).build();
  }

  DagPass() = delete;
  DagPass(optimizer::Config config, concrete_optimizer::Dag &dag)
      : config(config), dag(dag) {}
};

// Create an instance of the ConcreteOptimizerPass pass.
// A global pass result is communicated using `dags`.
// If `debug` is true, for each operation, the pass emits a
// remark containing the squared Minimal Arithmetic Noise Padding of
// the equivalent dot operation.
std::unique_ptr<mlir::Pass> createDagPass(optimizer::Config config,
                                          concrete_optimizer::Dag &dag) {
  return std::make_unique<optimizer::DagPass>(config, dag);
}

// Adds the composition rules to the
void applyCompositionRules(optimizer::Config config,
                           concrete_optimizer::Dag &dag) {

  if (config.composable) {
    dag.add_all_compositions();
    return;
  }
  for (auto rule : config.composition_rules) {
    dag.add_composition(rule.from_func, rule.from_pos, rule.to_func,
                        rule.to_pos);
  }
}

} // namespace optimizer
} // namespace concretelang
} // namespace mlir
