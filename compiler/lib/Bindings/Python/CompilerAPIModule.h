// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license
// information.

#ifndef CONCRETELANG_PYTHON_COMPILER_API_MODULE_H
#define CONCRETELANG_PYTHON_COMPILER_API_MODULE_H

#include <pybind11/pybind11.h>

namespace mlir {
namespace concretelang {
namespace python {

void populateCompilerAPISubmodule(pybind11::module &m);

} // namespace python
} // namespace concretelang
} // namespace mlir

#endif // CONCRETELANG_PYTHON_DIALECTMODULES_H