#ifndef CONCRETELANG_DIALECT_FHE_IR_FHE_INTERFACES
#define CONCRETELANG_DIALECT_FHE_IR_FHE_INTERFACES

include "mlir/IR/OpBase.td"

def FheIntegerInterface : TypeInterface<"FheIntegerInterface"> {
  let cppNamespace = "mlir::concretelang::FHE";

  let description = [{
    Interface for encapsulating the common properties of encrypted integer types.
  }];

  let methods = [
    InterfaceMethod<
      /*description=*/"Get bit-width of the integer.",
      /*retTy=*/"unsigned",
      /*methodName=*/"getWidth"
    >,
    InterfaceMethod<
      /*description=*/"Get whether the integer is signed.",
      /*retTy=*/"bool",
      /*methodName=*/"isSigned"
    >,
    InterfaceMethod<
      /*description=*/"Get whether the integer is unsigned.",
      /*retTy=*/"bool",
      /*methodName=*/"isUnsigned"
    >
  ];
}

def ConstantNoise : OpInterface<"ConstantNoise"> {
    let description = [{
        An operation which always has the same noise.
    }];

    let cppNamespace = "mlir::concretelang::FHE";
}

def ZeroNoise : OpInterface<"ZeroNoise"> {
    let description = [{
        An operation outputs a ciphertext with zero noise.
    }];

    let cppNamespace = "mlir::concretelang::FHE";
}

def AdditiveNoise : OpInterface<"AdditiveNoise"> {
    let description = [{
        An n-ary operation whose output noise is the unweighted sum of all input noises.
    }];

    let cppNamespace = "mlir::concretelang::FHE";
}

def MaxNoise : OpInterface<"MaxNoise"> {
    let description = [{
        An n-ary operation whose output noise is the max of all input noises.
    }];

    let cppNamespace = "mlir::concretelang::FHE";
}

def UnaryEint : OpInterface<"UnaryEint"> {
    let description = [{
        A unary operation on scalars, with the operand encrypted.
    }];

    let cppNamespace = "mlir::concretelang::FHE";

    let methods = [
        InterfaceMethod<
        /*description=*/"Calculate squared MANP",
        /*retTy=*/"llvm::APInt",
        /*methodName=*/"sqMANP",
        /*ins=*/(ins "llvm::APInt":$a),
        /*methodBody=*/"",
        /*defaultImplementation=*/"return a;"
        >,
        InterfaceMethod<
        /*description=*/"Get the underlying integer type of an operand",
        /*retTy=*/"mlir::Type",
        /*methodName=*/"operandIntType",
        /*ins=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          if (auto operandTy = dyn_cast<mlir::RankedTensorType>($_op->getOpOperand(0).get().getType())) {
            return operandTy.getElementType();
          } else return $_op->getOpOperand(0).get().getType();
        }]>
    ];
}

def Binary : OpInterface<"Binary"> {
    let description = [{
        An operation with two operands
    }];

    let cppNamespace = "mlir::concretelang::FHE";

    let methods = [
        InterfaceMethod<
        /*description=*/"Calculate squared MANP",
        /*retTy=*/"llvm::APInt",
        /*methodName=*/"sqMANP",
        /*ins=*/(ins "llvm::APInt":$a),
        /*methodBody=*/"",
        /*defaultImplementation=*/"return a;"
        >,
        InterfaceMethod<
        /*description=*/"Get the underlying integer type of an operand",
        /*retTy=*/"mlir::Type",
        /*methodName=*/"operandIntType",
        /*ins=*/(ins "unsigned":$opNum),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          if (auto operandTy = dyn_cast<mlir::RankedTensorType>($_op->getOpOperand(opNum).get().getType())) {
            return operandTy.getElementType();
          } else return $_op->getOpOperand(opNum).get().getType();
        }]>,
        InterfaceMethod<
        /*description=*/"Get the (largest) scalar value of an operand",
        /*retTy=*/"std::optional<llvm::APInt>",
        /*methodName=*/"operandMaxConstant",
        /*ins=*/(ins "unsigned":$opNum),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          if (auto cstOp = llvm::dyn_cast_or_null<mlir::arith::ConstantOp>($_op->
                                                        getOpOperand(opNum).get().getDefiningOp())) {
            if (auto operandTy = dyn_cast<mlir::RankedTensorType>($_op->
                                                        getOpOperand(opNum).get().getType())) {
              mlir::DenseIntElementsAttr denseVals =
                cstOp->template getAttrOfType<mlir::DenseIntElementsAttr>("value");
              return *(std::max_element(denseVals.begin(), denseVals.end(),
              [](llvm::APInt a, llvm::APInt b) {
                return a.ult(b);
              }));
            } else return cstOp->template getAttrOfType<mlir::IntegerAttr>("value").getValue();
          } else return {};
        }]>,
        InterfaceMethod<
        /*description=*/"Get clear operand tensor value ",
        /*retTy=*/"std::optional<mlir::detail::ElementsAttrRange<mlir::DenseElementsAttr::IntElementIterator>>",
        /*methodName=*/"opTensorConstant",
        /*ins=*/(ins "unsigned":$opNum),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          if (auto cstOp = llvm::dyn_cast_or_null<mlir::arith::ConstantOp>($_op->
                                                        getOpOperand(opNum).get().getDefiningOp()))
            return cstOp->template getAttrOfType<mlir::DenseIntElementsAttr>("value").template getValues<llvm::APInt>();
          else return {};
        }]>,
    ];
}

def BinaryEintInt : OpInterface<"BinaryEintInt", [Binary]> {
    let description = [{
        A binary operation on scalars, with the first operand encrypted and the
        second clear.
    }];

    let cppNamespace = "mlir::concretelang::FHE";

    let methods = [
        InterfaceMethod<
        /*description=*/"Get clear operand number",
        /*retTy=*/"unsigned",
        /*methodName=*/"getClearOperandNumber",
        /*ins=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/"return 1;"
        >
    ];
}

def BinaryIntEint : OpInterface<"BinaryIntEint", [Binary]> {
    let description = [{
        A binary operation on scalars, with the first operand clear and the
        second encrypted.
    }];

    let cppNamespace = "mlir::concretelang::FHE";

    let methods = [
        InterfaceMethod<
        /*description=*/"Get clear operand number",
        /*retTy=*/"unsigned",
        /*methodName=*/"getClearOperandNumber",
        /*ins=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/"return 0;"
        >
    ];
}

def BinaryEint : OpInterface<"BinaryEint"> {
    let description = [{
        A binary operation on scalars, with both operands encrypted.
    }];

    let cppNamespace = "mlir::concretelang::FHE";

    let methods = [
        InterfaceMethod<
        /*description=*/"Calculate squared MANP",
        /*retTy=*/"llvm::APInt",
        /*methodName=*/"sqMANP",
        /*ins=*/(ins "llvm::APInt":$a, "llvm::APInt":$b)
        >
    ];
}

#endif
