#include <cmath>

#include "EndToEndFixture.h"
#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/Support/Jit.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"

using mlir::concretelang::StreamStringError;

const mlir::concretelang::V0FHEConstraint defaultV0Constraints{1, 1};

// derived from https://stackoverflow.com/a/45869209
uint64_t solve_binomial_cdf_bigger_than(size_t n, double p_error,
                                        double p_mass) {
  // Solve: find k such that
  //   - binomial_cdf(k, n, p_error) >= p_mass
  //   - given n and p_error and p_mass
  // Notation:
  //   - n is the number of repetition,
  //   - p_error the probability of error,
  //   - p_mass is the mass probability of having a number of error <= k
  //   - k is a number of error that is returned
  // This returns the smallest threshold for error such that the number of error
  // is below or equal, with probability p_mass.
  using std::exp;
  using std::log;
  auto log_p_error = log(p_error);
  auto log_p_success = log(1.0 - p_error);
  auto cdf_k = 0.0;
  double log_pmf_k;
  // k in 0..n, isum and stop when the cumulative attained p_mass
  for (uint64_t k = 0; k < n; k++) {
    if (k == 0) {
      // start with n success
      log_pmf_k = n * log_p_success;
    } else {
      // add one error and remove one success
      log_pmf_k += log(n - k + 1) - log(k) + (log_p_error - log_p_success);
    }
    cdf_k += exp(log_pmf_k);
    if (cdf_k > p_mass) {
      return k;
    }
  }
  return n;
}

uint64_t TestErrorRate::too_high_error_count_threshold() {
  // Return the smallest threshold for error such that
  // the number of error is higher than this threshold
  // with probability FALSE_ALARM_RATE.

  // Examples:
  // n_rep=100, p_error=0.05 -> 16 error max  (ratio 3)
  // n_rep=100, p_error=0.01 -> 7 error max (ratio 7)
  // n_rep=100, p_error=0.001 -> 3 error max (ratio 30)
  // n_rep=10000, p_error=0.0001 -> 8 error max (ratio 8)
  // A high ratio indicate that the number of repetition should be increased.
  // A good ratio (but costly) is between 1 and 2.
  // A bad ratio can still detect most issues.
  // A good ratio will help to detect more precise calibration issue.
  double p_mass = 1.0 - TestErrorRate::FALSE_ALARM_RATE;
  return solve_binomial_cdf_bigger_than(this->nb_repetition, this->p_error,
                                        p_mass);
}

llvm::Expected<mlir::concretelang::LambdaArgument *>
scalarDescToLambdaArgument(ScalarDesc desc) {
  switch (desc.width) {
  case 8:
    return new mlir::concretelang::IntLambdaArgument<uint8_t>(desc.value);
  case 16:
    return new mlir::concretelang::IntLambdaArgument<uint16_t>(desc.value);
  case 32:
    return new mlir::concretelang::IntLambdaArgument<uint32_t>(desc.value);
  case 64:
    return new mlir::concretelang::IntLambdaArgument<uint64_t>(desc.value);
  }
  return StreamStringError("unsupported width of scalar value: ") << desc.width;
}

llvm::Expected<mlir::concretelang::LambdaArgument *>
TensorDescriptionToLambdaArgument(TensorDescription desc) {
  switch (desc.width) {
  case 8:;
    return new mlir::concretelang::TensorLambdaArgument<
        mlir::concretelang::IntLambdaArgument<uint8_t>>(
        std::vector<uint8_t>(desc.values.begin(), desc.values.end()),
        desc.shape);
  case 16:
    return new mlir::concretelang::TensorLambdaArgument<
        mlir::concretelang::IntLambdaArgument<uint16_t>>(
        std::vector<uint16_t>(desc.values.begin(), desc.values.end()),
        desc.shape);
  case 32:
    return new mlir::concretelang::TensorLambdaArgument<
        mlir::concretelang::IntLambdaArgument<uint32_t>>(
        std::vector<uint32_t>(desc.values.begin(), desc.values.end()),
        desc.shape);

  case 64:
    return new mlir::concretelang::TensorLambdaArgument<
        mlir::concretelang::IntLambdaArgument<uint64_t>>(desc.values,
                                                         desc.shape);
  }
  return StreamStringError("unsupported width of tensor value: ") << desc.width;
}

llvm::Expected<mlir::concretelang::LambdaArgument *>
valueDescriptionToLambdaArgument(ValueDescription desc) {
  switch (desc.tag) {
  case ValueDescription::SCALAR:
    return scalarDescToLambdaArgument(desc.scalar);
  case ValueDescription::TENSOR:
    return TensorDescriptionToLambdaArgument(desc.tensor);
  }
  return StreamStringError("unsupported value description");
}

llvm::Error checkResult(ScalarDesc &desc,
                        mlir::concretelang::LambdaArgument &res) {
  auto res64 = res.dyn_cast<mlir::concretelang::IntLambdaArgument<uint64_t>>();
  if (res64 == nullptr) {
    return StreamStringError("invocation result is not a scalar");
  }
  if (desc.value != res64->getValue()) {
    return StreamStringError("unexpected result value: got ")
           << res64->getValue() << " expected " << desc.value;
  }
  return llvm::Error::success();
}

template <typename UINT>
llvm::Error
checkTensorResult(TensorDescription &desc,
                  mlir::concretelang::TensorLambdaArgument<
                      mlir::concretelang::IntLambdaArgument<UINT>> *res) {
  if (!desc.shape.empty()) {
    auto resShape = res->getDimensions();
    if (desc.shape.size() != resShape.size()) {
      return StreamStringError("size of shape differs, got ")
             << resShape.size() << " expected " << desc.shape.size();
    }
    for (size_t i = 0; i < desc.shape.size(); i++) {
      if (resShape[i] != desc.shape[i]) {
        return StreamStringError("shape differs at pos ")
               << i << ", got " << resShape[i] << " expected " << desc.shape[i];
      }
    }
  }
  auto resValues = res->getValue();
  auto numElts = res->getNumElements();
  if (!numElts) {
    return numElts.takeError();
  }
  if (desc.values.size() != *numElts) {
    return StreamStringError("size of result differs, got ")
           << *numElts << " expected " << desc.values.size();
  }
  for (size_t i = 0; i < *numElts; i++) {
    if (resValues[i] != desc.values[i]) {
      return StreamStringError("result value differ at pos(")
             << i << "), got " << resValues[i] << " expected "
             << desc.values[i];
    }
  }

  return llvm::Error::success();
}

llvm::Error checkResult(TensorDescription &desc,
                        mlir::concretelang::LambdaArgument &res) {
  switch (desc.width) {
  case 8:
    return checkTensorResult<uint8_t>(
        desc, res.dyn_cast<mlir::concretelang::TensorLambdaArgument<
                  mlir::concretelang::IntLambdaArgument<uint8_t>>>());
  case 16:
    return checkTensorResult<uint16_t>(
        desc, res.dyn_cast<mlir::concretelang::TensorLambdaArgument<
                  mlir::concretelang::IntLambdaArgument<uint16_t>>>());
  case 32:
    return checkTensorResult<uint32_t>(
        desc, res.dyn_cast<mlir::concretelang::TensorLambdaArgument<
                  mlir::concretelang::IntLambdaArgument<uint32_t>>>());
  case 64:
    return checkTensorResult<uint64_t>(
        desc, res.dyn_cast<mlir::concretelang::TensorLambdaArgument<
                  mlir::concretelang::IntLambdaArgument<uint64_t>>>());
  default:
    return StreamStringError("Unsupported width");
  }
}

llvm::Error checkResult(ValueDescription &desc,
                        mlir::concretelang::LambdaArgument &res) {
  switch (desc.tag) {
  case ValueDescription::SCALAR:
    return checkResult(desc.scalar, res);
  case ValueDescription::TENSOR:
    return checkResult(desc.tensor, res);
  }
  assert(false);
}

template <> struct llvm::yaml::MappingTraits<ValueDescription> {
  static void mapping(IO &io, ValueDescription &desc) {
    auto keys = io.keys();
    if (std::find(keys.begin(), keys.end(), "scalar") != keys.end()) {
      io.mapRequired("scalar", desc.scalar.value);
      io.mapOptional("width", desc.scalar.width, 64);
      desc.tag = ValueDescription::SCALAR;
      return;
    }
    if (std::find(keys.begin(), keys.end(), "tensor") != keys.end()) {
      io.mapRequired("tensor", desc.tensor.values);
      io.mapOptional("width", desc.tensor.width, 64);
      io.mapRequired("shape", desc.tensor.shape);
      desc.tag = ValueDescription::TENSOR;
      return;
    }
    io.setError("Missing scalar or tensor key");
  }
};

LLVM_YAML_IS_SEQUENCE_VECTOR(ValueDescription)

template <> struct llvm::yaml::MappingTraits<TestDescription> {
  static void mapping(IO &io, TestDescription &desc) {
    io.mapOptional("inputs", desc.inputs);
    io.mapOptional("outputs", desc.outputs);
  }
};

LLVM_YAML_IS_SEQUENCE_VECTOR(TestDescription)

template <> struct llvm::yaml::MappingTraits<TestErrorRate> {
  static void mapping(IO &io, TestErrorRate &desc) {
    io.mapRequired("p-error", desc.p_error);
    io.mapRequired("nb-repetition", desc.nb_repetition);
  }
};

LLVM_YAML_IS_SEQUENCE_VECTOR(TestErrorRate)

template <> struct llvm::yaml::MappingTraits<EndToEndDesc> {
  static void mapping(IO &io, EndToEndDesc &desc) {
    io.mapRequired("description", desc.description);
    io.mapRequired("program", desc.program);
    io.mapRequired("tests", desc.tests);

    bool use_default_fhe_constraints = false;
    io.mapOptional("use_default_fhe_constraints", use_default_fhe_constraints);

    if (use_default_fhe_constraints)
      desc.v0Constraint = defaultV0Constraints;

    std::vector<int64_t> v0parameter;
    io.mapOptional("v0-parameter", v0parameter);
    if (!v0parameter.empty()) {
      if (v0parameter.size() != 7) {
        io.setError("v0-parameter expect to be a list 7 elemnts "
                    "[glweDimension, logPolynomialSize, nSmall, brLevel, "
                    "brLobBase, ksLevel, ksLogBase]");
      }
      desc.v0Parameter = mlir::concretelang::V0Parameter(
          v0parameter[0], v0parameter[1], v0parameter[2], v0parameter[3],
          v0parameter[4], v0parameter[5], v0parameter[6]);
    }
    std::vector<int64_t> v0constraint;
    io.mapOptional("v0-constraint", v0constraint);
    if (!v0constraint.empty()) {
      if (v0constraint.size() != 2) {
        io.setError("v0-constraint expect to be a list 2 elemnts "
                    "[p, norm2]");
      }
      desc.v0Constraint = mlir::concretelang::V0FHEConstraint();
      desc.v0Constraint->p = v0constraint[0];
      desc.v0Constraint->norm2 = v0constraint[1];
    }
    mlir::concretelang::LargeIntegerParameter largeInterger;
    io.mapOptional("large-integer-crt-decomposition",
                   largeInterger.crtDecomposition);
    if (!largeInterger.crtDecomposition.empty()) {
      desc.largeIntegerParameter = largeInterger;
    }
    io.mapOptional("test-error-rates", desc.test_error_rates);
  }
};

LLVM_YAML_IS_DOCUMENT_LIST_VECTOR(EndToEndDesc)

std::vector<EndToEndDesc> loadEndToEndDesc(std::string path) {
  std::ifstream file(path);

  if (!file.good()) {
    std::cerr << "Could not read yaml file: " << path << std::endl;
    assert(false);
  }

  std::string content((std::istreambuf_iterator<char>(file)),
                      (std::istreambuf_iterator<char>()));

  llvm::yaml::Input yin(content);

  // Parse the YAML file
  std::vector<EndToEndDesc> desc;
  yin >> desc;

  // Check for error
  if (yin.error()) { // .error() displays the beginning of the error message
    std::cerr << "In yaml file: " << path << "\n";
    assert(false);
  }
  return desc;
}
