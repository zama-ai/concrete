#ifndef ZAMALANG_PYTHON_DIALECTMODULES_H
#define ZAMALANG_PYTHON_DIALECTMODULES_H

#include <pybind11/pybind11.h>

namespace mlir {
namespace zamalang {
namespace python {

void populateDialectHLFHESubmodule(pybind11::module &m);

} // namespace python
} // namespace zamalang
} // namespace mlir

#endif // ZAMALANG_PYTHON_DIALECTMODULES_H