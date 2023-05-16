#ifndef CONCRETELANG_CLIENTLIB_DATA_H
#define CONCRETELANG_CLIENTLIB_DATA_H

#include "boost/outcome.h"

#include "concretelang/ClientLib/ClientParameters.h"
#include "concretelang/ClientLib/Types.h"
#include "concretelang/Common/Error.h"

namespace concretelang {
namespace clientlib {

using concretelang::error::StringError;

class Data {
public:
  Data(CircuitGate description, ScalarOrTensorData data);

  Data(Data &other) = delete;
  Data(Data &&other)
      : description{other.description}, data{std::move(other.data)} {};

  bool isEncrypted() const { return this->description.isEncrypted(); }
  bool isClear() const { return !this->isEncrypted(); }

  bool isScalar() const { return this->description.shape.dimensions.empty(); }
  bool isTensor() const { return !this->isScalar(); }

  bool isSigned() const {
    return this->isEncrypted() ? this->description.encryption->encoding.isSigned
                               : true;
  }
  bool isUnsigned() const { return !this->isSigned(); }

  const std::vector<int64_t> &shape() const {
    return this->description.shape.dimensions;
  }
  size_t ndims() const { return this->shape().size(); }

  static outcome::checked<std::unique_ptr<Data>, StringError>
  deserialize(std::istream &istream);

  outcome::checked<void, StringError> serialize(std::ostream &ostream);

private:
  CircuitGate description;
  ScalarOrTensorData data;
};

} // namespace clientlib
} // namespace concretelang

#endif
