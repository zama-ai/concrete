#ifndef END_TO_END_FIXTURE_H
#define END_TO_END_FIXTURE_H

#include "concretelang/Common/Values.h"
#include "concretelang/Support/V0Parameters.h"
#include <fstream>
#include <string>
#include <vector>

using concretelang::values::Tensor;
using concretelang::values::Value;

struct ValueDescription {
  ValueDescription() : value(nullptr) {}
  ValueDescription(const ValueDescription &other) : value(other.value) {}

  template <typename T> void setValue(T value) {
    auto scalarVal = Tensor<T>(value);
    this->value = std::make_shared<Value>(scalarVal);
  }

  template <typename T>
  void setValue(std::vector<T> values, std::vector<int64_t> shape) {
    auto convertedShape = std::vector<size_t>();
    convertedShape.resize(shape.size());
    for (size_t i = 0; i < shape.size(); i++) {
      convertedShape[i] = (size_t)shape[i];
    }
    auto tensorVal = Tensor<T>(values, convertedShape);
    this->value = std::make_shared<Value>(tensorVal);
  }

  const Value &getValue() const {
    assert(this->value != nullptr);
    return *value;
  }

protected:
  std::shared_ptr<Value> value;
};

struct TestDescription {
  std::vector<ValueDescription> inputs;
  std::vector<ValueDescription> outputs;
};

struct TestErrorRate {
  double global_p_error;
  uint64_t nb_repetition;
  // rate at which a valid code will make the test fail due to bad luck
  const double FALSE_ALARM_RATE = 0.00001;
  uint64_t too_high_error_count_threshold();
};

struct EndToEndDesc {
  std::string description;
  std::string program;
  std::optional<double> p_error; // force the test in local p-error
  std::vector<TestDescription> tests;
  std::optional<mlir::concretelang::V0Parameter> v0Parameter;
  std::optional<mlir::concretelang::V0FHEConstraint> v0Constraint;
  concrete_optimizer::Encoding encoding;
  std::optional<mlir::concretelang::LargeIntegerParameter>
      largeIntegerParameter;
  std::vector<TestErrorRate> test_error_rates;
};

struct EndToEndDescFile {
  std::string path;
  std::vector<EndToEndDesc> descriptions;
};

llvm::Error checkResult(ValueDescription &desc, Value &res);

/// Unserialize from the given path a list of a end to end description file.
std::vector<EndToEndDesc> loadEndToEndDesc(std::string path);

#endif
