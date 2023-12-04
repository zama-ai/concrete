#ifndef CONCRETELANG_INTERFACES_BATCHABLEINTERFACE
#define CONCRETELANG_INTERFACES_BATCHABLEINTERFACE

include "mlir/IR/OpBase.td"

def BatchableOpInterface : OpInterface<"BatchableOpInterface"> {
  let description = [{
      Interface for operations processing a scalar that can be batched
    if invoked multiple times with different, independent operands.
  }];
  let cppNamespace = "::mlir::concretelang";

  let methods = [
    InterfaceMethod<[{
        Return the number of batching schemes for the operation.
      }],
      /*retTy=*/"unsigned",
      /*methodName=*/"getNumBatchingVariants",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return 1;
      }]
    >,
    InterfaceMethod<[{
        Return the scalar operands that can be batched in a tensor to
        be passed to the corresponding batched operation.
      }],
      /*retTy=*/"::llvm::MutableArrayRef<::mlir::OpOperand>",
      /*methodName=*/"getBatchableOperands",
      /*args=*/(ins "unsigned":$variant),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        llvm_unreachable("getBatchableOperands not implemented");
      }]
    >,
    InterfaceMethod<[{
        Create the batched operation and return it as a value.
      }],
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"createBatchedOperation",
      /*args=*/(ins "unsigned":$variant,
                    "::mlir::ImplicitLocOpBuilder&":$builder,
                    "::mlir::ValueRange":$batchedOperands,
                    "::mlir::ValueRange":$hoistedNonBatchableOperands),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        llvm_unreachable("createBatchedOperation not implemented");
      }]
    >
  ];
}

#endif // CONCRETELANG_INTERFACES_BATCHABLEINTERFACE
