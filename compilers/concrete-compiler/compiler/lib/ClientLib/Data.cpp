#include "concretelang/ClientLib/Data.h"
#include "concretelang/ClientLib/Serializers.h"

using concretelang::clientlib::Data;
using concretelang::error::StringError;

Data::Data(CircuitGate description, ScalarOrTensorData data)
    : description{description}, data{std::move(data)} {}

outcome::checked<std::unique_ptr<Data>, StringError>
Data::deserialize(std::istream &istream) {
  size_t descriptionSize = 0;
  readSize(istream, descriptionSize);

  if (istream.fail()) {
    return StringError("cannot read cleartext description size");
  }
  assert(descriptionSize != 0);

  auto buffer = std::vector<char>(descriptionSize + 1);
  istream.read(&buffer[0], descriptionSize);
  buffer[descriptionSize] = '\0';

  const auto descriptionString = std::string(buffer.begin(), buffer.end());
  llvm::Expected<CircuitGate> description =
      llvm::json::parse<CircuitGate>(descriptionString);

  if (!description) {
    return StringError("cannot read cleartext description");
  }

  const std::vector<int64_t> &expectedSizes = description->shape.dimensions;
  outcome::checked<ScalarOrTensorData, StringError> data =
      unserializeScalarOrTensorData(expectedSizes, istream);

  if (!data) {
    return StringError("cannot read cleartext data");
  }

  return std::make_unique<Data>(description.get(), std::move(data.value()));
}

outcome::checked<void, StringError> Data::serialize(std::ostream &ostream) {
  if (incorrectMode(ostream)) {
    return StringError("ostream should be in binary mode");
  }

  std::string descriptionString;
  llvm::raw_string_ostream descriptionBuffer(descriptionString);

  auto descriptionJson = llvm::json::Value(this->description);
  descriptionBuffer << descriptionJson;

  size_t descriptionSize = descriptionString.size();
  writeSize(ostream, descriptionSize);
  ostream << descriptionString;

  serializeScalarOrTensorData(this->data, ostream);
  if (ostream.fail()) {
    return StringError("cannot write data to ostream");
  }

  return outcome::success();
}
