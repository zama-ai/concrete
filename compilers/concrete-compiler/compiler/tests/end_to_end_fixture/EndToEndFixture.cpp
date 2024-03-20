#include <cmath>

#include "EndToEndFixture.h"
#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/Support/Error.h"
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
  return solve_binomial_cdf_bigger_than(this->nb_repetition,
                                        this->global_p_error, p_mass);
}

llvm::Error checkResult(ValueDescription &desc, Value &res) {

  if (!(desc.getValue() == res)) {
    // Todo -> Make a more informative error.
    return StreamStringError("expect ")
           << desc.getValue().toString() << ", got " << res.toString();
  } else {
    return llvm::Error::success();
  }
};

template <typename T> struct ReadScalar {
  static void read(llvm::yaml::IO &io, ValueDescription &desc) {
    T v;
    io.mapRequired("scalar", v);
    desc.setValue(v);
  }
};

static void readScalar(llvm::yaml::IO &io, ValueDescription &desc,
                       unsigned int width, bool sign) {
  if (width > 32) {
    if (sign)
      ReadScalar<int64_t>::read(io, desc);
    else
      ReadScalar<uint64_t>::read(io, desc);
  } else if (width > 16) {
    if (sign)
      ReadScalar<int32_t>::read(io, desc);
    else
      ReadScalar<uint32_t>::read(io, desc);
  } else if (width > 8) {
    if (sign)
      ReadScalar<int16_t>::read(io, desc);
    else
      ReadScalar<uint16_t>::read(io, desc);
  } else {
    if (sign)
      ReadScalar<int8_t>::read(io, desc);
    else
      ReadScalar<uint8_t>::read(io, desc);
  }
}

template <typename T> struct ReadTensor {
  static void read(llvm::yaml::IO &io, ValueDescription &desc) {
    std::vector<T> values;
    std::vector<int64_t> dimensions;

    io.mapRequired("shape", dimensions);
    io.mapRequired("tensor", values);

    desc.setValue(values, dimensions);
  }
};

static void readTensor(llvm::yaml::IO &io, ValueDescription &desc,
                       unsigned int width, bool sign) {
  if (width > 32) {
    if (sign)
      ReadTensor<int64_t>::read(io, desc);
    else
      ReadTensor<uint64_t>::read(io, desc);
  } else if (width > 16) {
    if (sign)
      ReadTensor<int32_t>::read(io, desc);
    else
      ReadTensor<uint32_t>::read(io, desc);
  } else if (width > 8) {
    if (sign)
      ReadTensor<int16_t>::read(io, desc);
    else
      ReadTensor<uint16_t>::read(io, desc);
  } else {
    if (sign)
      ReadTensor<int8_t>::read(io, desc);
    else
      ReadTensor<uint8_t>::read(io, desc);
  }
}

template <> struct llvm::yaml::MappingTraits<ValueDescription> {
  static void mapping(IO &io, ValueDescription &desc) {
    unsigned int width;
    bool sign;

    auto keys = io.keys();
    if (std::find(keys.begin(), keys.end(), "scalar") != keys.end()) {
      io.mapOptional("width", width, 64);
      io.mapOptional("signed", sign, false);

      if (width > 64)
        io.setError("Scalar values must have 64 bits or less");

      readScalar(io, desc, width, sign);
      return;
    }
    if (std::find(keys.begin(), keys.end(), "tensor") != keys.end()) {
      io.mapOptional("width", width, 64);

      if (width > 64)
        io.setError("Scalar values must have 64 bits or less");

      io.mapOptional("signed", sign, false);

      readTensor(io, desc, width, sign);
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
    io.mapRequired("global-p-error", desc.global_p_error);
    io.mapRequired("nb-repetition", desc.nb_repetition);
  }
};

LLVM_YAML_IS_SEQUENCE_VECTOR(TestErrorRate)

template <> struct llvm::yaml::MappingTraits<EndToEndDesc> {
  static void mapping(IO &io, EndToEndDesc &desc) {
    io.mapRequired("description", desc.description);
    io.mapRequired("program", desc.program);
    io.mapOptional("p-error", desc.p_error);
    io.mapRequired("tests", desc.tests);
    bool use_default_fhe_constraints = false;
    io.mapOptional("use_default_fhe_constraints", use_default_fhe_constraints);

    if (use_default_fhe_constraints)
      desc.v0Constraint = defaultV0Constraints;

    std::vector<int64_t> v0parameter;
    io.mapOptional("v0-parameter", v0parameter);
    if (!v0parameter.empty()) {
      if (v0parameter.size() != 7) {
        io.setError("v0-parameter expect to be a list 7 elements "
                    "[glweDimension, logPolynomialSize, nSmall, brLevel, "
                    "brLobBase, ksLevel, ksLogBase]");
      }
      desc.v0Parameter = {(size_t)v0parameter[0], (size_t)v0parameter[1],
                          (size_t)v0parameter[2], (size_t)v0parameter[3],
                          (size_t)v0parameter[4], (size_t)v0parameter[5],
                          (size_t)v0parameter[6], std::nullopt};
    }
    std::vector<int64_t> v0constraint;
    io.mapOptional("v0-constraint", v0constraint);
    if (!v0constraint.empty()) {
      if (v0constraint.size() != 2) {
        io.setError("v0-constraint expect to be a list 2 elements "
                    "[p, norm2]");
      }
      desc.v0Constraint = mlir::concretelang::V0FHEConstraint();
      desc.v0Constraint->p = v0constraint[0];
      desc.v0Constraint->norm2 = v0constraint[1];
    }
    std::string str_encoding = "auto";
    io.mapOptional("encoding", str_encoding);
    if (str_encoding == "auto") {
      desc.encoding = concrete_optimizer::Encoding::Auto;
    } else if (str_encoding == "native") {
      desc.encoding = concrete_optimizer::Encoding::Native;
    } else if (str_encoding == "crt") {
      desc.encoding = concrete_optimizer::Encoding::Crt;
    } else {
      io.setError("encoding can only be native or crt");
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
