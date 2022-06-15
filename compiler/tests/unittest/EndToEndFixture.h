#ifndef END_TO_END_FIXTURE_H
#define END_TO_END_FIXTURE_H

#include <fstream>
#include <string>
#include <vector>

#include "end_to_end_jit_test.h"

typedef uint8_t ValueWidth;
struct TensorDescription {
  std::vector<int64_t> shape;
  std::vector<uint64_t> values;
  ValueWidth width;
};
struct ScalarDesc {
  uint64_t value;
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

struct EndToEndDesc {
  std::string description;
  std::string program;
  std::vector<TestDescription> tests;
};

llvm::Expected<mlir::concretelang::LambdaArgument *>
scalarDescToLambdaArgument(ScalarDesc desc);

llvm::Expected<mlir::concretelang::LambdaArgument *>
valueDescriptionToLambdaArgument(ValueDescription desc);

llvm::Error checkResult(ScalarDesc &desc,
                        mlir::concretelang::LambdaArgument *res);

llvm::Error checkResult(ValueDescription &desc,
                        mlir::concretelang::LambdaArgument &res);

std::vector<EndToEndDesc> loadEndToEndDesc(std::string path);

std::string printEndToEndDesc(const testing::TestParamInfo<EndToEndDesc> desc);

#endif
