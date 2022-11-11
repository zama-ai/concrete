//! FHELinalg dialect module

use crate::mlir::*;

pub fn create_fhelinalg_add_eint_op(
    context: MlirContext,
    lhs: MlirValue,
    rhs: MlirValue,
    result_type: MlirType,
) -> MlirOperation {
    create_op(
        context,
        "FHELinalg.add_eint",
        &[lhs, rhs],
        [result_type].as_slice(),
        &[],
        false,
    )
}

pub fn create_fhelinalg_add_eint_int_op(
    context: MlirContext,
    lhs: MlirValue,
    rhs: MlirValue,
    result_type: MlirType,
) -> MlirOperation {
    create_op(
        context,
        "FHELinalg.add_eint_int",
        &[lhs, rhs],
        [result_type].as_slice(),
        &[],
        false,
    )
}

pub fn create_fhelinalg_sub_eint_op(
    context: MlirContext,
    lhs: MlirValue,
    rhs: MlirValue,
    result_type: MlirType,
) -> MlirOperation {
    create_op(
        context,
        "FHELinalg.sub_eint",
        &[lhs, rhs],
        [result_type].as_slice(),
        &[],
        false,
    )
}

pub fn create_fhelinalg_sub_eint_int_op(
    context: MlirContext,
    lhs: MlirValue,
    rhs: MlirValue,
    result_type: MlirType,
) -> MlirOperation {
    create_op(
        context,
        "FHELinalg.sub_eint_int",
        &[lhs, rhs],
        [result_type].as_slice(),
        &[],
        false,
    )
}

pub fn create_fhelinalg_sub_int_eint_op(
    context: MlirContext,
    lhs: MlirValue,
    rhs: MlirValue,
    result_type: MlirType,
) -> MlirOperation {
    create_op(
        context,
        "FHELinalg.sub_int_eint",
        &[lhs, rhs],
        [result_type].as_slice(),
        &[],
        false,
    )
}

pub fn create_fhelinalg_negate_eint_op(
    context: MlirContext,
    eint_tensor: MlirValue,
) -> MlirOperation {
    unsafe {
        let results = [mlirValueGetType(eint_tensor)];
        // infer result type from operands
        create_op(
            context,
            "FHELinalg.neg_eint",
            &[eint_tensor],
            results.as_slice(),
            &[],
            false,
        )
    }
}

pub fn create_fhelinalg_mul_eint_int_op(
    context: MlirContext,
    lhs: MlirValue,
    rhs: MlirValue,
    result_type: MlirType,
) -> MlirOperation {
    create_op(
        context,
        "FHELinalg.mul_eint_int",
        &[lhs, rhs],
        [result_type].as_slice(),
        &[],
        false,
    )
}

pub fn create_fhelinalg_apply_lut_op(
    context: MlirContext,
    eint_tensor: MlirValue,
    lut: MlirValue,
    result_type: MlirType,
) -> MlirOperation {
    create_op(
        context,
        "FHELinalg.apply_lookup_table",
        &[eint_tensor, lut],
        [result_type].as_slice(),
        &[],
        false,
    )
}

pub fn create_fhelinalg_apply_multi_lut_op(
    context: MlirContext,
    eint_tensor: MlirValue,
    lut: MlirValue,
    result_type: MlirType,
) -> MlirOperation {
    create_op(
        context,
        "FHELinalg.apply_multi_lookup_table",
        &[eint_tensor, lut],
        [result_type].as_slice(),
        &[],
        false,
    )
}

pub fn create_fhelinalg_apply_mapped_lut_op(
    context: MlirContext,
    eint_tensor: MlirValue,
    lut: MlirValue,
    map: MlirValue,
    result_type: MlirType,
) -> MlirOperation {
    create_op(
        context,
        "FHELinalg.apply_mapped_lookup_table",
        &[eint_tensor, lut, map],
        [result_type].as_slice(),
        &[],
        false,
    )
}

pub fn create_fhelinalg_dot_eint_int_op(
    context: MlirContext,
    lhs: MlirValue,
    rhs: MlirValue,
    result_type: MlirType,
) -> MlirOperation {
    create_op(
        context,
        "FHELinalg.dot_eint_int",
        &[lhs, rhs],
        [result_type].as_slice(),
        &[],
        false,
    )
}

pub fn create_fhelinalg_matmul_eint_int_op(
    context: MlirContext,
    lhs: MlirValue,
    rhs: MlirValue,
    result_type: MlirType,
) -> MlirOperation {
    create_op(
        context,
        "FHELinalg.matmul_eint_int",
        &[lhs, rhs],
        [result_type].as_slice(),
        &[],
        false,
    )
}

pub fn create_fhelinalg_matmul_int_eint_op(
    context: MlirContext,
    lhs: MlirValue,
    rhs: MlirValue,
    result_type: MlirType,
) -> MlirOperation {
    create_op(
        context,
        "FHELinalg.matmul_int_eint",
        &[lhs, rhs],
        [result_type].as_slice(),
        &[],
        false,
    )
}

pub fn create_fhelinalg_sum_op(
    context: MlirContext,
    eint_tensor: MlirValue,
    axes: Option<MlirNamedAttribute>,
    keep_dims: Option<MlirNamedAttribute>,
    result_type: MlirType,
) -> MlirOperation {
    let mut attrs: Vec<MlirNamedAttribute> = Vec::new();
    if axes.is_some() {
        attrs.push(axes.unwrap());
    }
    if keep_dims.is_some() {
        attrs.push(keep_dims.unwrap());
    }
    create_op(
        context,
        "FHELinalg.sum",
        &[eint_tensor],
        [result_type].as_slice(),
        attrs.as_slice(),
        false,
    )
}

pub fn create_fhelinalg_concat_op(
    context: MlirContext,
    eint_tensor: MlirValue,
    axis: Option<MlirNamedAttribute>,
    result_type: MlirType,
) -> MlirOperation {
    let mut attrs: Vec<MlirNamedAttribute> = Vec::new();
    if axis.is_some() {
        attrs.push(axis.unwrap());
    }
    create_op(
        context,
        "FHELinalg.concat",
        &[eint_tensor],
        [result_type].as_slice(),
        &attrs,
        false,
    )
}

pub fn create_fhelinalg_conv2d_op(
    context: MlirContext,
    input: MlirValue,
    weight: MlirValue,
    bias: Option<MlirValue>,
    padding: Option<MlirNamedAttribute>,
    strides: Option<MlirNamedAttribute>,
    dilations: Option<MlirNamedAttribute>,
    group: Option<MlirNamedAttribute>,
    result_type: MlirType,
) -> MlirOperation {
    let mut operands = Vec::new();
    operands.push(input);
    operands.push(weight);
    if bias.is_some() {
        operands.push(bias.unwrap());
    }
    let mut attrs = Vec::new();
    if padding.is_some() {
        attrs.push(padding.unwrap());
    }
    if strides.is_some() {
        attrs.push(strides.unwrap());
    }
    if dilations.is_some() {
        attrs.push(dilations.unwrap());
    }
    if group.is_some() {
        attrs.push(group.unwrap());
    }
    create_op(
        context,
        "FHELinalg.conv2d",
        &operands,
        [result_type].as_slice(),
        &attrs,
        false,
    )
}

pub fn create_fhelinalg_transpose_op(
    context: MlirContext,
    eint_tensor: MlirValue,
    axes: Option<MlirNamedAttribute>,
    result_type: MlirType,
) -> MlirOperation {
    let mut attrs: Vec<MlirNamedAttribute> = Vec::new();
    if axes.is_some() {
        attrs.push(axes.unwrap());
    }
    create_op(
        context,
        "FHELinalg.transpose",
        &[eint_tensor],
        [result_type].as_slice(),
        attrs.as_slice(),
        false,
    )
}

pub fn create_fhelinalg_from_element_op(context: MlirContext, element: MlirValue) -> MlirOperation {
    unsafe {
        let location = mlirLocationUnknownGet(context);
        let shape: [i64; 1] = [1];
        let result_type = mlirRankedTensorTypeGetChecked(
            location,
            1,
            shape.as_ptr(),
            mlirValueGetType(element),
            mlirAttributeGetNull(),
        );
        create_op(
            context,
            "FHELinalg.from_element",
            &[element],
            [result_type].as_slice(),
            &[],
            false,
        )
    }
}

pub fn create_fhelinalg_to_signed_op(
    context: MlirContext,
    eint_tensor: MlirValue,
) -> MlirOperation {
    unsafe {
        let results = [mlirValueGetType(eint_tensor)];
        // infer result type from operands
        create_op(
            context,
            "FHELinalg.to_signed",
            &[eint_tensor],
            results.as_slice(),
            &[],
            false,
        )
    }
}

pub fn create_fhelinalg_to_unsigned_op(
    context: MlirContext,
    esint_tensor: MlirValue,
) -> MlirOperation {
    unsafe {
        let results = [mlirValueGetType(esint_tensor)];
        // infer result type from operands
        create_op(
            context,
            "FHELinalg.to_unsigned",
            &[esint_tensor],
            results.as_slice(),
            &[],
            false,
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::fhe::*;

    #[test]
    fn test_fhelinalg_func() {
        unsafe {
            let context = mlirContextCreate();
            mlirRegisterAllDialects(context);

            // register the FHELinalg dialect
            let fhe_handle = mlirGetDialectHandle__fhe__();
            mlirDialectHandleLoadDialect(fhe_handle, context);

            // register the FHELinalg dialect
            let fhelinalg_handle = mlirGetDialectHandle__fhelinalg__();
            mlirDialectHandleLoadDialect(fhelinalg_handle, context);

            // create a 5-bit eint tensor type
            let eint_or_error = fheEncryptedIntegerTypeGetChecked(context, 5);
            assert!(!eint_or_error.isError);
            let eint = eint_or_error.type_;
            let shape: [i64; 2] = [6, 73];
            let location = mlirLocationUnknownGet(context);
            let eint_tensor = mlirRankedTensorTypeGetChecked(
                location,
                2,
                shape.as_ptr(),
                eint,
                mlirAttributeGetNull(),
            );

            // set input/output types of the FHE circuit
            let func_input_types = [eint_tensor, eint_tensor];
            let func_output_types = [eint_tensor];

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
            let add_eint_op =
                create_fhelinalg_add_eint_op(context, func_args[0], func_args[1], eint_tensor);
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
  func.func @main(%arg0: tensor<6x73x!FHE.eint<5>>, %arg1: tensor<6x73x!FHE.eint<5>>) -> tensor<6x73x!FHE.eint<5>> {
    %0 = \"FHELinalg.add_eint\"(%arg0, %arg1) : (tensor<6x73x!FHE.eint<5>>, tensor<6x73x!FHE.eint<5>>) -> tensor<6x73x!FHE.eint<5>>
    return %0 : tensor<6x73x!FHE.eint<5>>
  }
}
";
            assert_eq!(printed_module, expected_module);
        }
    }

    #[test]
    fn test_zero_tensor_op() {
        unsafe {
            let context = mlirContextCreate();
            mlirRegisterAllDialects(context);

            // register the FHELinalg dialect
            let fhe_handle = mlirGetDialectHandle__fhe__();
            mlirDialectHandleLoadDialect(fhe_handle, context);

            // register the FHELinalg dialect
            let fhelinalg_handle = mlirGetDialectHandle__fhelinalg__();
            mlirDialectHandleLoadDialect(fhelinalg_handle, context);

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
}
