//! FHE dialect module

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

pub fn create_fhe_to_signed_op(context: MlirContext, eint: MlirValue) -> MlirOperation {
    unsafe {
        let results = [mlirValueGetType(eint)];
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
        let results = [mlirValueGetType(esint)];
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
            let printed_eint = super::print_mlir_type_to_string(eint);
            let expected_eint = "!FHE.eint<5>";
            assert_eq!(printed_eint, expected_eint);
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
}
