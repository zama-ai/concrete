#ifndef END_TO_END_FIXTURE_H
#define END_TO_END_FIXTURE_H

#include <fstream>
#include <string>
#include <vector>

#include "concretelang/Support/JITSupport.h"

typedef uint8_t ValueWidth;
struct TensorDescription {
  std::vector<int64_t> shape;
  std::vector<uint64_t> values;
  ValueWidth width;
};
struct ScalarDesc {
  int64_t value;
  ValueWidth width;
};

struct ValueDescription {
  template <typename T> static ValueDescription get(T value) {
    ValueDescription desc;
    desc.tag = ValueDescription::SCALAR;
    desc.scalar.value = (uint64_t)value;
    desc.scalar.width = sizeof(value) * 8;
    return desc;
  }
  enum { SCALAR, TENSOR } tag;

  ScalarDesc scalar;
  TensorDescription tensor;
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
  bool loopParallelize;
  bool dataflowParallelize;
  bool asyncOffload;
};

llvm::Expected<mlir::concretelang::LambdaArgument *>
scalarDescToLambdaArgument(ScalarDesc desc);

llvm::Expected<mlir::concretelang::LambdaArgument *>
valueDescriptionToLambdaArgument(ValueDescription desc);

llvm::Error checkResult(ScalarDesc &desc,
                        mlir::concretelang::LambdaArgument *res);

llvm::Error checkResult(ValueDescription &desc,
                        mlir::concretelang::LambdaArgument &res);

/// Unserialize from the given path a list of a end to end description file.
std::vector<EndToEndDesc> loadEndToEndDesc(std::string path);

#endif
