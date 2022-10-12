#ifndef END_TO_END_FIXTURE_H
#define END_TO_END_FIXTURE_H

#include <fstream>
#include <string>
#include <vector>

#include "concretelang/ClientLib/Types.h"
#include "concretelang/Support/JITSupport.h"

struct ValueDescription {
  ValueDescription() : value(nullptr) {}
  ValueDescription(const ValueDescription &other) : value(other.value) {}

  template <typename T> void setValue(T value) {
    this->value =
        std::make_shared<mlir::concretelang::IntLambdaArgument<T>>(value);
  }

  template <typename T>
  void setValue(std::vector<T> &&value, llvm::ArrayRef<int64_t> shape) {
    this->value = std::make_shared<mlir::concretelang::TensorLambdaArgument<
        mlir::concretelang::IntLambdaArgument<T>>>(value, shape);
  }

  const mlir::concretelang::LambdaArgument &getValue() const {
    assert(this->value != nullptr);
    return *value;
  }

protected:
  std::shared_ptr<mlir::concretelang::LambdaArgument> value;
};

struct TestDescription {
  std::vector<ValueDescription> inputs;
  std::vector<ValueDescription> outputs;
};

struct TestErrorRate {
  double p_error;
  uint64_t nb_repetition;
  // rate at which a valid code will make the test fail due to bad luck
  const double FALSE_ALARM_RATE = 0.00001;
  uint64_t too_high_error_count_threshold();
};

struct EndToEndDesc {
  std::string description;
  std::string program;
  std::vector<TestDescription> tests;
  llvm::Optional<mlir::concretelang::V0Parameter> v0Parameter;
  llvm::Optional<mlir::concretelang::V0FHEConstraint> v0Constraint;
  llvm::Optional<mlir::concretelang::LargeIntegerParameter>
      largeIntegerParameter;
  std::vector<TestErrorRate> test_error_rates;
};

llvm::Error checkResult(ValueDescription &desc,
                        mlir::concretelang::LambdaArgument &res);

/// Unserialize from the given path a list of a end to end description file.
std::vector<EndToEndDesc> loadEndToEndDesc(std::string path);

#endif
