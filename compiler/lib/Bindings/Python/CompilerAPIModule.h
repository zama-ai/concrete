// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.

#ifndef ZAMALANG_PYTHON_COMPILER_API_MODULE_H
#define ZAMALANG_PYTHON_COMPILER_API_MODULE_H

#include <pybind11/pybind11.h>

namespace mlir {
namespace zamalang {
namespace python {

void populateCompilerAPISubmodule(pybind11::module &m);

} // namespace python
} // namespace zamalang
} // namespace mlir

#endif // ZAMALANG_PYTHON_DIALECTMODULES_H