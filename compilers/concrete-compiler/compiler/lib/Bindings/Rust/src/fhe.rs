//! FHE dialect module

use crate::mlir::ffi::*;
use crate::mlir::*;

pub fn create_fhe_add_eint_op(
    context: MlirContext,
    lhs: MlirValue,
    rhs: MlirValue,
) -> MlirOperation {
    unsafe {
        let results = [mlirValueGetType(lhs)];
        // infer result type from operands
        create_op(
            context,
            "FHE.add_eint",
            &[lhs, rhs],
            results.as_slice(),
            &[],
            false,
        )
    }
}

pub fn create_fhe_add_eint_int_op(
    context: MlirContext,
    lhs: MlirValue,
    rhs: MlirValue,
) -> MlirOperation {
    unsafe {
        let results = [mlirValueGetType(lhs)];
        // infer result type from operands
        create_op(
            context,
            "FHE.add_eint_int",
            &[lhs, rhs],
            results.as_slice(),
            &[],
            false,
        )
    }
}

pub fn create_fhe_sub_eint_op(
    context: MlirContext,
    lhs: MlirValue,
    rhs: MlirValue,
) -> MlirOperation {
    unsafe {
        let results = [mlirValueGetType(lhs)];
        // infer result type from operands
        create_op(
            context,
            "FHE.sub_eint",
            &[lhs, rhs],
            results.as_slice(),
            &[],
            false,
        )
    }
}

pub fn create_fhe_sub_eint_int_op(
    context: MlirContext,
    lhs: MlirValue,
    rhs: MlirValue,
) -> MlirOperation {
    unsafe {
        let results = [mlirValueGetType(lhs)];
        // infer result type from operands
        create_op(
            context,
            "FHE.sub_eint_int",
            &[lhs, rhs],
            results.as_slice(),
            &[],
            false,
        )
    }
}

pub fn create_fhe_sub_int_eint_op(
    context: MlirContext,
    lhs: MlirValue,
    rhs: MlirValue,
) -> MlirOperation {
    unsafe {
        let results = [mlirValueGetType(rhs)];
        // infer result type from operands
        create_op(
            context,
            "FHE.sub_int_eint",
            &[lhs, rhs],
            results.as_slice(),
            &[],
            false,
        )
    }
}

pub fn create_fhe_negate_eint_op(context: MlirContext, eint: MlirValue) -> MlirOperation {
    unsafe {
        let results = [mlirValueGetType(eint)];
        // infer result type from operands
        create_op(
            context,
            "FHE.neg_eint",
            &[eint],
            results.as_slice(),
            &[],
            false,
        )
    }
}

pub fn create_fhe_mul_eint_int_op(
    context: MlirContext,
    lhs: MlirValue,
    rhs: MlirValue,
) -> MlirOperation {
    unsafe {
        let results = [mlirValueGetType(lhs)];
        // infer result type from operands
        create_op(
            context,
            "FHE.mul_eint_int",
            &[lhs, rhs],
            results.as_slice(),
            &[],
            false,
        )
    }
}

pub fn create_fhe_apply_lut_op(
    context: MlirContext,
    eint: MlirValue,
    lut: MlirValue,
    result_type: MlirType,
) -> MlirOperation {
    create_op(
        context,
        "FHE.apply_lookup_table",
        &[eint, lut],
        [result_type].as_slice(),
        &[],
        false,
    )
}

#[derive(Debug)]
pub enum FHEError {
    InvalidFHEType,
    InvalidWidth,
}

pub fn convert_eint_to_esint_type(
    context: MlirContext,
    eint_type: MlirType,
) -> Result<MlirType, FHEError> {
    unsafe {
        let width = fheTypeIntegerWidthGet(eint_type);
        if width == 0 {
            return Err(FHEError::InvalidFHEType);
        }
        let type_or_error = fheEncryptedSignedIntegerTypeGetChecked(context, width);
        if type_or_error.isError {
            Err(FHEError::InvalidWidth)
        } else {
            Ok(type_or_error.type_)
        }
    }
}

pub fn convert_esint_to_eint_type(
    context: MlirContext,
    esint_type: MlirType,
) -> Result<MlirType, FHEError> {
    unsafe {
        let width = fheTypeIntegerWidthGet(esint_type);
        if width == 0 {
            return Err(FHEError::InvalidFHEType);
        }
        let type_or_error = fheEncryptedIntegerTypeGetChecked(context, width);
        if type_or_error.isError {
            Err(FHEError::InvalidWidth)
        } else {
            Ok(type_or_error.type_)
        }
    }
}

pub fn create_fhe_to_signed_op(context: MlirContext, eint: MlirValue) -> MlirOperation {
    unsafe {
        let results = [convert_eint_to_esint_type(context, mlirValueGetType(eint)).unwrap()];
        // infer result type from operands
        create_op(
            context,
            "FHE.to_signed",
            &[eint],
            results.as_slice(),
            &[],
            false,
        )
    }
}

pub fn create_fhe_to_unsigned_op(context: MlirContext, esint: MlirValue) -> MlirOperation {
    unsafe {
        let results = [convert_esint_to_eint_type(context, mlirValueGetType(esint)).unwrap()];
        // infer result type from operands
        create_op(
            context,
            "FHE.to_unsigned",
            &[esint],
            results.as_slice(),
            &[],
            false,
        )
    }
}

pub fn create_fhe_zero_eint_op(context: MlirContext, result_type: MlirType) -> MlirOperation {
    create_op(
        context,
        "FHE.zero",
        &[],
        [result_type].as_slice(),
        &[],
        false,
    )
}

pub fn create_fhe_zero_eint_tensor_op(
    context: MlirContext,
    result_type: MlirType,
) -> MlirOperation {
    create_op(
        context,
        "FHE.zero_tensor",
        &[],
        [result_type].as_slice(),
        &[],
        false,
    )
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_invalid_fhe_eint_type() {
        unsafe {
            let context = mlirContextCreate();
            mlirRegisterAllDialects(context);
            let invalid_eint = fheEncryptedIntegerTypeGetChecked(context, 0);
            assert!(invalid_eint.isError);
        }
    }

    #[test]
    fn test_valid_fhe_eint_type() {
        unsafe {
            let context = mlirContextCreate();
            mlirRegisterAllDialects(context);
            let fhe_handle = mlirGetDialectHandle__fhe__();
            mlirDialectHandleLoadDialect(fhe_handle, context);
            let eint_or_error = fheEncryptedIntegerTypeGetChecked(context, 5);
            assert!(!eint_or_error.isError);
            let eint = eint_or_error.type_;
            assert!(fheTypeIsAnEncryptedIntegerType(eint));
            assert!(!fheTypeIsAnEncryptedSignedIntegerType(eint));
            assert_eq!(fheTypeIntegerWidthGet(eint), 5);
            let printed_eint = super::print_mlir_type_to_string(eint);
            let expected_eint = "!FHE.eint<5>";
            assert_eq!(printed_eint, expected_eint);
        }
    }

    #[test]
    fn test_valid_fhe_esint_type() {
        unsafe {
            let context = mlirContextCreate();
            mlirRegisterAllDialects(context);
            let fhe_handle = mlirGetDialectHandle__fhe__();
            mlirDialectHandleLoadDialect(fhe_handle, context);
            let esint_or_error = fheEncryptedSignedIntegerTypeGetChecked(context, 5);
            assert!(!esint_or_error.isError);
            let esint = esint_or_error.type_;
            assert!(fheTypeIsAnEncryptedSignedIntegerType(esint));
            assert!(!fheTypeIsAnEncryptedIntegerType(esint));
            assert_eq!(fheTypeIntegerWidthGet(esint), 5);
            let printed_esint = super::print_mlir_type_to_string(esint);
            let expected_esint = "!FHE.esint<5>";
            assert_eq!(printed_esint, expected_esint);
        }
    }

    #[test]
    fn test_fhe_func() {
        unsafe {
            let context = mlirContextCreate();
            mlirRegisterAllDialects(context);

            // register the FHE dialect
            let fhe_handle = mlirGetDialectHandle__fhe__();
            mlirDialectHandleLoadDialect(fhe_handle, context);

            // create a 5-bit eint type
            let eint_or_error = fheEncryptedIntegerTypeGetChecked(context, 5);
            assert!(!eint_or_error.isError);
            let eint = eint_or_error.type_;

            // set input/output types of the FHE circuit
            let func_input_types = [eint, eint];
            let func_output_types = [eint];

            // create the func operation
            let func_op = create_func_with_block(
                context,
                "main",
                func_input_types.as_slice(),
                func_output_types.as_slice(),
            );
            let func_block = mlirRegionGetFirstBlock(mlirOperationGetFirstRegion(func_op));
            let func_args = [
                mlirBlockGetArgument(func_block, 0),
                mlirBlockGetArgument(func_block, 1),
            ];

            // create an FHE add_eint op and append it to the function block
            let add_eint_op = create_fhe_add_eint_op(context, func_args[0], func_args[1]);
            mlirBlockAppendOwnedOperation(func_block, add_eint_op);

            // create ret operation and append it to the block
            let ret_op = create_ret_op(context, mlirOperationGetResult(add_eint_op, 0));
            mlirBlockAppendOwnedOperation(func_block, ret_op);

            // create module to hold the previously created function
            let location = mlirLocationUnknownGet(context);
            let module = mlirModuleCreateEmpty(location);
            mlirBlockAppendOwnedOperation(mlirModuleGetBody(module), func_op);

            let printed_module =
                super::print_mlir_operation_to_string(mlirModuleGetOperation(module));
            let expected_module = "\
module {
  func.func @main(%arg0: !FHE.eint<5>, %arg1: !FHE.eint<5>) -> !FHE.eint<5> {
    %0 = \"FHE.add_eint\"(%arg0, %arg1) : (!FHE.eint<5>, !FHE.eint<5>) -> !FHE.eint<5>
    return %0 : !FHE.eint<5>
  }
}
";
            assert_eq!(printed_module, expected_module);
        }
    }

    #[test]
    fn test_zero_op() {
        unsafe {
            let context = mlirContextCreate();
            mlirRegisterAllDialects(context);

            // register the FHE dialect
            let fhe_handle = mlirGetDialectHandle__fhe__();
            mlirDialectHandleLoadDialect(fhe_handle, context);

            // create a 6-bit eint type
            let eint_or_error = fheEncryptedIntegerTypeGetChecked(context, 6);
            assert!(!eint_or_error.isError);
            let eint6_type = eint_or_error.type_;

            let zero_op = create_fhe_zero_eint_op(context, eint6_type);
            let printed_op = print_mlir_operation_to_string(zero_op);
            let expected_op = "%0 = \"FHE.zero\"() : () -> !FHE.eint<6>\n";
            assert_eq!(printed_op, expected_op);
        }
    }

    #[test]
    fn test_zero_tensor_op() {
        unsafe {
            let context = mlirContextCreate();
            mlirRegisterAllDialects(context);

            // register the FHE dialect
            let fhe_handle = mlirGetDialectHandle__fhe__();
            mlirDialectHandleLoadDialect(fhe_handle, context);

            // create a 4-bit eint tensor type
            let eint_or_error = fheEncryptedIntegerTypeGetChecked(context, 4);
            assert!(!eint_or_error.isError);
            let eint = eint_or_error.type_;
            let shape: [i64; 3] = [60, 66, 73];
            let location = mlirLocationUnknownGet(context);
            let eint_tensor = mlirRankedTensorTypeGetChecked(
                location,
                3,
                shape.as_ptr(),
                eint,
                mlirAttributeGetNull(),
            );

            let zero_op = create_fhe_zero_eint_tensor_op(context, eint_tensor);
            let printed_op = print_mlir_operation_to_string(zero_op);
            let expected_op = "%0 = \"FHE.zero_tensor\"() : () -> tensor<60x66x73x!FHE.eint<4>>\n";
            assert_eq!(printed_op, expected_op);
        }
    }

    #[test]
    fn test_add_eint_op() {
        unsafe {
            let context = mlirContextCreate();
            mlirRegisterAllDialects(context);

            // register the FHE dialect
            let fhe_handle = mlirGetDialectHandle__fhe__();
            mlirDialectHandleLoadDialect(fhe_handle, context);

            // create a 6-bit eint type
            let eint_or_error = fheEncryptedIntegerTypeGetChecked(context, 6);
            assert!(!eint_or_error.isError);
            let eint6_type = eint_or_error.type_;

            // create module for ops
            let location = mlirLocationUnknownGet(context);
            let module = mlirModuleCreateEmpty(location);
            let main_block = mlirModuleGetBody(module);
            // create an encrypted integer via a zero_op
            let zero_op = create_fhe_zero_eint_op(context, eint6_type);
            mlirBlockAppendOwnedOperation(main_block, zero_op);
            let eint_value = mlirOperationGetResult(zero_op, 0);
            // add eint with itself
            let add_eint_op = create_fhe_add_eint_op(context, eint_value, eint_value);
            mlirBlockAppendOwnedOperation(main_block, add_eint_op);

            let printed_op = print_mlir_operation_to_string(add_eint_op);
            let expected_op =
                "%1 = \"FHE.add_eint\"(%0, %0) : (!FHE.eint<6>, !FHE.eint<6>) -> !FHE.eint<6>";
            assert_eq!(printed_op, expected_op);
        }
    }

    #[test]
    fn test_add_eint_int_op() {
        unsafe {
            let context = mlirContextCreate();
            mlirRegisterAllDialects(context);

            // register the FHE dialect
            let fhe_handle = mlirGetDialectHandle__fhe__();
            mlirDialectHandleLoadDialect(fhe_handle, context);

            // create a 6-bit eint type
            let eint_or_error = fheEncryptedIntegerTypeGetChecked(context, 6);
            assert!(!eint_or_error.isError);
            let eint6_type = eint_or_error.type_;

            // create module for ops
            let location = mlirLocationUnknownGet(context);
            let module = mlirModuleCreateEmpty(location);
            let main_block = mlirModuleGetBody(module);
            // create an encrypted integer via a zero_op
            let zero_op = create_fhe_zero_eint_op(context, eint6_type);
            mlirBlockAppendOwnedOperation(main_block, zero_op);
            let eint_value = mlirOperationGetResult(zero_op, 0);
            // create an int via a constant op
            let cst_op = create_constant_int_op(context, 73, 7);
            mlirBlockAppendOwnedOperation(main_block, cst_op);
            let int_value = mlirOperationGetResult(cst_op, 0);
            // add eint int
            let add_eint_int_op = create_fhe_add_eint_int_op(context, eint_value, int_value);
            mlirBlockAppendOwnedOperation(main_block, add_eint_int_op);

            let printed_op = print_mlir_operation_to_string(add_eint_int_op);
            let expected_op =
                "%1 = \"FHE.add_eint_int\"(%0, %c-55_i7) : (!FHE.eint<6>, i7) -> !FHE.eint<6>";
            assert_eq!(printed_op, expected_op);
        }
    }

    #[test]
    fn test_sub_eint_op() {
        unsafe {
            let context = mlirContextCreate();
            mlirRegisterAllDialects(context);

            // register the FHE dialect
            let fhe_handle = mlirGetDialectHandle__fhe__();
            mlirDialectHandleLoadDialect(fhe_handle, context);

            // create a 6-bit eint type
            let eint_or_error = fheEncryptedIntegerTypeGetChecked(context, 6);
            assert!(!eint_or_error.isError);
            let eint6_type = eint_or_error.type_;

            // create module for ops
            let location = mlirLocationUnknownGet(context);
            let module = mlirModuleCreateEmpty(location);
            let main_block = mlirModuleGetBody(module);
            // create an encrypted integer via a zero_op
            let zero_op = create_fhe_zero_eint_op(context, eint6_type);
            mlirBlockAppendOwnedOperation(main_block, zero_op);
            let eint_value = mlirOperationGetResult(zero_op, 0);
            // sub eint with itself
            let sub_eint_op = create_fhe_sub_eint_op(context, eint_value, eint_value);
            mlirBlockAppendOwnedOperation(main_block, sub_eint_op);

            let printed_op = print_mlir_operation_to_string(sub_eint_op);
            let expected_op =
                "%1 = \"FHE.sub_eint\"(%0, %0) : (!FHE.eint<6>, !FHE.eint<6>) -> !FHE.eint<6>";
            assert_eq!(printed_op, expected_op);
        }
    }

    #[test]
    fn test_sub_eint_int_op() {
        unsafe {
            let context = mlirContextCreate();
            mlirRegisterAllDialects(context);

            // register the FHE dialect
            let fhe_handle = mlirGetDialectHandle__fhe__();
            mlirDialectHandleLoadDialect(fhe_handle, context);

            // create a 6-bit eint type
            let eint_or_error = fheEncryptedIntegerTypeGetChecked(context, 6);
            assert!(!eint_or_error.isError);
            let eint6_type = eint_or_error.type_;

            // create module for ops
            let location = mlirLocationUnknownGet(context);
            let module = mlirModuleCreateEmpty(location);
            let main_block = mlirModuleGetBody(module);
            // create an encrypted integer via a zero_op
            let zero_op = create_fhe_zero_eint_op(context, eint6_type);
            mlirBlockAppendOwnedOperation(main_block, zero_op);
            let eint_value = mlirOperationGetResult(zero_op, 0);
            // create an int via a constant op
            let cst_op = create_constant_int_op(context, 73, 7);
            mlirBlockAppendOwnedOperation(main_block, cst_op);
            let int_value = mlirOperationGetResult(cst_op, 0);
            // sub eint int
            let sub_eint_int_op = create_fhe_sub_eint_int_op(context, eint_value, int_value);
            mlirBlockAppendOwnedOperation(main_block, sub_eint_int_op);

            let printed_op = print_mlir_operation_to_string(sub_eint_int_op);
            let expected_op =
                "%1 = \"FHE.sub_eint_int\"(%0, %c-55_i7) : (!FHE.eint<6>, i7) -> !FHE.eint<6>";
            assert_eq!(printed_op, expected_op);
        }
    }

    #[test]
    fn test_sub_int_eint_op() {
        unsafe {
            let context = mlirContextCreate();
            mlirRegisterAllDialects(context);

            // register the FHE dialect
            let fhe_handle = mlirGetDialectHandle__fhe__();
            mlirDialectHandleLoadDialect(fhe_handle, context);

            // create a 6-bit eint type
            let eint_or_error = fheEncryptedIntegerTypeGetChecked(context, 6);
            assert!(!eint_or_error.isError);
            let eint6_type = eint_or_error.type_;

            // create module for ops
            let location = mlirLocationUnknownGet(context);
            let module = mlirModuleCreateEmpty(location);
            let main_block = mlirModuleGetBody(module);
            // create an encrypted integer via a zero_op
            let zero_op = create_fhe_zero_eint_op(context, eint6_type);
            mlirBlockAppendOwnedOperation(main_block, zero_op);
            let eint_value = mlirOperationGetResult(zero_op, 0);
            // create an int via a constant op
            let cst_op = create_constant_int_op(context, 73, 7);
            mlirBlockAppendOwnedOperation(main_block, cst_op);
            let int_value = mlirOperationGetResult(cst_op, 0);
            // sub int eint
            let sub_eint_int_op = create_fhe_sub_int_eint_op(context, int_value, eint_value);
            mlirBlockAppendOwnedOperation(main_block, sub_eint_int_op);

            let printed_op = print_mlir_operation_to_string(sub_eint_int_op);
            let expected_op =
                "%1 = \"FHE.sub_int_eint\"(%c-55_i7, %0) : (i7, !FHE.eint<6>) -> !FHE.eint<6>";
            assert_eq!(printed_op, expected_op);
        }
    }

    #[test]
    fn test_negate_eint_op() {
        unsafe {
            let context = mlirContextCreate();
            mlirRegisterAllDialects(context);

            // register the FHE dialect
            let fhe_handle = mlirGetDialectHandle__fhe__();
            mlirDialectHandleLoadDialect(fhe_handle, context);

            // create a 6-bit eint type
            let eint_or_error = fheEncryptedIntegerTypeGetChecked(context, 6);
            assert!(!eint_or_error.isError);
            let eint6_type = eint_or_error.type_;

            // create module for ops
            let location = mlirLocationUnknownGet(context);
            let module = mlirModuleCreateEmpty(location);
            let main_block = mlirModuleGetBody(module);
            // create an encrypted integer via a zero_op
            let zero_op = create_fhe_zero_eint_op(context, eint6_type);
            mlirBlockAppendOwnedOperation(main_block, zero_op);
            let eint_value = mlirOperationGetResult(zero_op, 0);
            // negate eint
            let neg_eint_op = create_fhe_negate_eint_op(context, eint_value);
            mlirBlockAppendOwnedOperation(main_block, neg_eint_op);

            let printed_op = print_mlir_operation_to_string(neg_eint_op);
            let expected_op = "%1 = \"FHE.neg_eint\"(%0) : (!FHE.eint<6>) -> !FHE.eint<6>";
            assert_eq!(printed_op, expected_op);
        }
    }

    #[test]
    fn test_mul_eint_int_op() {
        unsafe {
            let context = mlirContextCreate();
            mlirRegisterAllDialects(context);

            // register the FHE dialect
            let fhe_handle = mlirGetDialectHandle__fhe__();
            mlirDialectHandleLoadDialect(fhe_handle, context);

            // create a 6-bit eint type
            let eint_or_error = fheEncryptedIntegerTypeGetChecked(context, 6);
            assert!(!eint_or_error.isError);
            let eint6_type = eint_or_error.type_;

            // create module for ops
            let location = mlirLocationUnknownGet(context);
            let module = mlirModuleCreateEmpty(location);
            let main_block = mlirModuleGetBody(module);
            // create an encrypted integer via a zero_op
            let zero_op = create_fhe_zero_eint_op(context, eint6_type);
            mlirBlockAppendOwnedOperation(main_block, zero_op);
            let eint_value = mlirOperationGetResult(zero_op, 0);
            // create an int via a constant op
            let cst_op = create_constant_int_op(context, 73, 7);
            mlirBlockAppendOwnedOperation(main_block, cst_op);
            let int_value = mlirOperationGetResult(cst_op, 0);
            // mul eint int
            let mul_eint_int_op = create_fhe_mul_eint_int_op(context, eint_value, int_value);
            mlirBlockAppendOwnedOperation(main_block, mul_eint_int_op);

            let printed_op = print_mlir_operation_to_string(mul_eint_int_op);
            let expected_op =
                "%1 = \"FHE.mul_eint_int\"(%0, %c-55_i7) : (!FHE.eint<6>, i7) -> !FHE.eint<6>";
            assert_eq!(printed_op, expected_op);
        }
    }

    #[test]
    fn test_to_signed_op() {
        unsafe {
            let context = mlirContextCreate();
            mlirRegisterAllDialects(context);

            // register the FHE dialect
            let fhe_handle = mlirGetDialectHandle__fhe__();
            mlirDialectHandleLoadDialect(fhe_handle, context);

            // create a 6-bit eint type
            let eint_or_error = fheEncryptedIntegerTypeGetChecked(context, 6);
            assert!(!eint_or_error.isError);
            let eint6_type = eint_or_error.type_;

            // create module for ops
            let location = mlirLocationUnknownGet(context);
            let module = mlirModuleCreateEmpty(location);
            let main_block = mlirModuleGetBody(module);
            // create an encrypted integer via a zero_op
            let zero_op = create_fhe_zero_eint_op(context, eint6_type);
            mlirBlockAppendOwnedOperation(main_block, zero_op);
            let eint_value = mlirOperationGetResult(zero_op, 0);
            // to signed
            let to_signed_op = create_fhe_to_signed_op(context, eint_value);
            mlirBlockAppendOwnedOperation(main_block, to_signed_op);

            let printed_op = print_mlir_operation_to_string(to_signed_op);
            let expected_op = "%1 = \"FHE.to_signed\"(%0) : (!FHE.eint<6>) -> !FHE.esint<6>";
            assert_eq!(printed_op, expected_op);
        }
    }

    #[test]
    fn test_to_unsigned_op() {
        unsafe {
            let context = mlirContextCreate();
            mlirRegisterAllDialects(context);

            // register the FHE dialect
            let fhe_handle = mlirGetDialectHandle__fhe__();
            mlirDialectHandleLoadDialect(fhe_handle, context);

            // create a 6-bit esint type
            let esint_or_error = fheEncryptedSignedIntegerTypeGetChecked(context, 6);
            assert!(!esint_or_error.isError);
            let esint6_type = esint_or_error.type_;

            // create module for ops
            let location = mlirLocationUnknownGet(context);
            let module = mlirModuleCreateEmpty(location);
            let main_block = mlirModuleGetBody(module);
            // create an encrypted integer via a zero_op
            let zero_op = create_fhe_zero_eint_op(context, esint6_type);
            mlirBlockAppendOwnedOperation(main_block, zero_op);
            let esint_value = mlirOperationGetResult(zero_op, 0);
            // to unsigned
            let to_unsigned_op = create_fhe_to_unsigned_op(context, esint_value);
            mlirBlockAppendOwnedOperation(main_block, to_unsigned_op);

            let printed_op = print_mlir_operation_to_string(to_unsigned_op);
            let expected_op = "%1 = \"FHE.to_unsigned\"(%0) : (!FHE.esint<6>) -> !FHE.eint<6>";
            assert_eq!(printed_op, expected_op);
        }
    }

    #[test]
    fn test_apply_lut_op() {
        unsafe {
            let context = mlirContextCreate();
            mlirRegisterAllDialects(context);

            // register the FHE dialect
            let fhe_handle = mlirGetDialectHandle__fhe__();
            mlirDialectHandleLoadDialect(fhe_handle, context);

            // create a 6-bit eint type
            let eint_or_error = fheEncryptedIntegerTypeGetChecked(context, 6);
            assert!(!eint_or_error.isError);
            let eint6_type = eint_or_error.type_;

            // create module for ops
            let location = mlirLocationUnknownGet(context);
            let module = mlirModuleCreateEmpty(location);
            let main_block = mlirModuleGetBody(module);
            // create an encrypted integer via a zero_op
            let zero_op = create_fhe_zero_eint_op(context, eint6_type);
            mlirBlockAppendOwnedOperation(main_block, zero_op);
            let eint_value = mlirOperationGetResult(zero_op, 0);
            // create an lut
            let table: [i64; 64] = [0; 64];
            let constant_lut_op = create_constant_flat_tensor_op(context, &table, 64);
            mlirBlockAppendOwnedOperation(main_block, constant_lut_op);
            let lut = mlirOperationGetResult(constant_lut_op, 0);
            // LUT op
            let apply_lut_op = create_fhe_apply_lut_op(context, eint_value, lut, eint6_type);
            mlirBlockAppendOwnedOperation(main_block, apply_lut_op);

            let printed_op = print_mlir_operation_to_string(apply_lut_op);
            let expected_op = "%1 = \"FHE.apply_lookup_table\"(%0, %cst) : (!FHE.eint<6>, tensor<64xi64>) -> !FHE.eint<6>";
            assert_eq!(printed_op, expected_op);
        }
    }
}
