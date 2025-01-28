#ifndef CONCRETELANG_DIALECT_GLWE_IR_PARAMETERVARIABLEDETAIS_H
#define CONCRETELANG_DIALECT_GLWE_IR_PARAMETERVARIABLEDETAIS_H

#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/StorageUniquer.h"
struct ParameterVariableStorage : public mlir::StorageUniquer::BaseStorage {
  mlir::MLIRContext *context;
};
#endif // CONCRETELANG_DIALECT_GLWE_IR_PARAMETERVARIABLEDETAIS_H