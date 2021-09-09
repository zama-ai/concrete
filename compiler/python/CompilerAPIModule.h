#ifndef ZAMALANG_PYTHON_COMPILER_API_MODULE_H
#define ZAMALANG_PYTHON_COMPILER_API_MODULE_H

#include <pybind11/pybind11.h>

namespace zamalang {
namespace python {

// Frontend object to abstract the different types of possible arguments,
// namely, integers, and tensors.
class ExecutionArgument {
public:
  // There are two possible underlying types for the execution argument, either
  // and int, or a tensor
  bool isTensor() { return isTensorArg; }
  bool isInt() { return !isTensorArg; }

  uint8_t *getTensorArgument() { return tensorArg.data(); }

  size_t getTensorSize() { return tensorArg.size(); }

  uint64_t getIntegerArgument() { return intArg; }

  // Create an execution argument from an integer
  static std::shared_ptr<ExecutionArgument> create(uint64_t arg) {
    return std::shared_ptr<ExecutionArgument>(new ExecutionArgument(arg));
  }
  // Create an execution argument from a tensor
  static std::shared_ptr<ExecutionArgument> create(std::vector<uint8_t> arg) {
    return std::shared_ptr<ExecutionArgument>(new ExecutionArgument(arg));
  }

private:
  ExecutionArgument(int arg)
      : isTensorArg(false), intArg(arg) {}

  ExecutionArgument(std::vector<uint8_t> tensor)
      : isTensorArg(true), tensorArg(tensor) {}

  uint64_t intArg;
  std::vector<uint8_t> tensorArg;
  bool isTensorArg;
};

void populateCompilerAPISubmodule(pybind11::module &m);

} // namespace python
} // namespace zamalang

#endif // ZAMALANG_PYTHON_DIALECTMODULES_H