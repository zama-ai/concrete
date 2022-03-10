#include "EndToEndFixture.h"
#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/Support/Jit.h"
#include "concretelang/Support/JitCompilerEngine.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"

using mlir::concretelang::StreamStringError;

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
           << res64->getValue() << "expected " << desc.value;
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

std::string printEndToEndDesc(const testing::TestParamInfo<EndToEndDesc> desc) {
  return desc.param.description;
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

LLVM_YAML_IS_SEQUENCE_VECTOR(ValueDescription);

template <> struct llvm::yaml::MappingTraits<TestDescription> {
  static void mapping(IO &io, TestDescription &desc) {
    io.mapOptional("inputs", desc.inputs);
    io.mapOptional("outputs", desc.outputs);
  }
};

LLVM_YAML_IS_SEQUENCE_VECTOR(TestDescription);

template <> struct llvm::yaml::MappingTraits<EndToEndDesc> {
  static void mapping(IO &io, EndToEndDesc &desc) {
    io.mapRequired("description", desc.description);
    io.mapRequired("program", desc.program);
    io.mapRequired("tests", desc.tests);
  }
};

LLVM_YAML_IS_DOCUMENT_LIST_VECTOR(EndToEndDesc)

std::vector<EndToEndDesc> loadEndToEndDesc(std::string path) {
  std::ifstream file(path);
  std::string content((std::istreambuf_iterator<char>(file)),
                      (std::istreambuf_iterator<char>()));

  llvm::yaml::Input yin(content);

  // Parse the YAML file
  std::vector<EndToEndDesc> desc;
  yin >> desc;

  // Check for error
  if (yin.error())
    assert(false && "cannot parse doc");
  return desc;
}