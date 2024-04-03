// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CONVERSION_RTOPCONVERTER_H_
#define CONCRETELANG_CONVERSION_RTOPCONVERTER_H_

#include "concretelang/Conversion/Utils/Legality.h"
#include "concretelang/Conversion/Utils/ReinstantiatingOpTypeConversion.h"
#include "concretelang/Dialect/RT/IR/RTOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace concretelang {

inline void
populateWithRTTypeConverterPatterns(mlir::RewritePatternSet &patterns,
                                    mlir::ConversionTarget &target,
                                    mlir::TypeConverter &converter) {
  patterns.add<
      mlir::concretelang::TypeConvertingReinstantiationPattern<
          mlir::concretelang::RT::DataflowTaskOp>,
      mlir::concretelang::TypeConvertingReinstantiationPattern<
          mlir::concretelang::RT::DataflowYieldOp>,
      mlir::concretelang::TypeConvertingReinstantiationPattern<
          mlir::concretelang::RT::MakeReadyFutureOp>,
      mlir::concretelang::TypeConvertingReinstantiationPattern<
          mlir::concretelang::RT::AwaitFutureOp>,
      mlir::concretelang::TypeConvertingReinstantiationPattern<
          mlir::concretelang::RT::CreateAsyncTaskOp, true>,
      mlir::concretelang::TypeConvertingReinstantiationPattern<
          mlir::concretelang::RT::BuildReturnPtrPlaceholderOp>,
      mlir::concretelang::TypeConvertingReinstantiationPattern<
          mlir::concretelang::RT::DerefWorkFunctionArgumentPtrPlaceholderOp>,
      mlir::concretelang::TypeConvertingReinstantiationPattern<
          mlir::concretelang::RT::DerefReturnPtrPlaceholderOp>,
      mlir::concretelang::TypeConvertingReinstantiationPattern<
          mlir::concretelang::RT::WorkFunctionReturnOp>,
      mlir::concretelang::TypeConvertingReinstantiationPattern<
          mlir::concretelang::RT::RegisterTaskWorkFunctionOp>>(
      patterns.getContext(), converter);

  mlir::concretelang::addDynamicallyLegalTypeOp<
      mlir::concretelang::RT::DataflowTaskOp>(target, converter);
  mlir::concretelang::addDynamicallyLegalTypeOp<
      mlir::concretelang::RT::DataflowYieldOp>(target, converter);
  mlir::concretelang::addDynamicallyLegalTypeOp<
      mlir::concretelang::RT::MakeReadyFutureOp>(target, converter);
  mlir::concretelang::addDynamicallyLegalTypeOp<
      mlir::concretelang::RT::AwaitFutureOp>(target, converter);
  mlir::concretelang::addDynamicallyLegalTypeOp<
      mlir::concretelang::RT::CreateAsyncTaskOp>(target, converter);
  mlir::concretelang::addDynamicallyLegalTypeOp<
      mlir::concretelang::RT::BuildReturnPtrPlaceholderOp>(target, converter);
  mlir::concretelang::addDynamicallyLegalTypeOp<
      mlir::concretelang::RT::DerefWorkFunctionArgumentPtrPlaceholderOp>(
      target, converter);
  mlir::concretelang::addDynamicallyLegalTypeOp<
      mlir::concretelang::RT::DerefReturnPtrPlaceholderOp>(target, converter);
  mlir::concretelang::addDynamicallyLegalTypeOp<
      mlir::concretelang::RT::WorkFunctionReturnOp>(target, converter);
  mlir::concretelang::addDynamicallyLegalTypeOp<
      mlir::concretelang::RT::RegisterTaskWorkFunctionOp>(target, converter);
}
} // namespace concretelang
} // namespace mlir

#endif
