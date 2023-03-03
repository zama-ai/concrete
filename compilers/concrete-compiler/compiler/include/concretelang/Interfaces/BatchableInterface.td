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
        Return the scalar operand that can be batched in a tensor to
        be passed to the corresponding batched operation.
      }],
      /*retTy=*/"::mlir::OpOperand&",
      /*methodName=*/"getBatchableOperand",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        llvm_unreachable("getBatchableOperand not implemented");
      }]
    >,
    InterfaceMethod<[{
        Return all operands that cannot be batched.
      }],
      /*retTy=*/"::mlir::OperandRange",
      /*methodName=*/"getNonBatchableOperands",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        llvm_unreachable("getNonBatchableOperands not implemented");
      }]
    >,
    InterfaceMethod<[{
        Create the batched operation and return it as a value.
      }],
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"createBatchedOperation",
      /*args=*/(ins "::mlir::ImplicitLocOpBuilder&":$builder,
                    "::mlir::Value":$batchedOperands),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        llvm_unreachable("createBatchedOperation not implemented");
      }]
    >
  ];
}

#endif // CONCRETELANG_INTERFACES_BATCHABLEINTERFACE
