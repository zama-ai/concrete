//! FHELinalg dialect module

use crate::{
    fhe::{convert_eint_to_esint_type, convert_esint_to_eint_type},
    mlir::ffi::*,
    mlir::*,
};
use std::ffi::CString;

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
    axes: Option<&[i64]>,
    keep_dims: Option<bool>,
    result_type: MlirType,
) -> MlirOperation {
    unsafe {
        let mut attrs: Vec<MlirNamedAttribute> = Vec::new();
        match axes {
            Some(value) => {
                let axes_str = CString::new("axes").unwrap();
                let axes_attrs: Vec<MlirAttribute> = value
                    .into_iter()
                    .map(|value| mlirIntegerAttrGet(mlirIntegerTypeGet(context, 64), *value))
                    .collect();
                attrs.push(mlirNamedAttributeGet(
                    mlirIdentifierGet(context, mlirStringRefCreateFromCString(axes_str.as_ptr())),
                    mlirArrayAttrGet(
                        context,
                        value.len().try_into().unwrap(),
                        axes_attrs.as_ptr(),
                    ),
                ));
            }
            None => (),
        }
        match keep_dims {
            Some(value) => {
                let keep_dims_str = CString::new("keep_dims").unwrap();
                attrs.push(mlirNamedAttributeGet(
                    mlirIdentifierGet(
                        context,
                        mlirStringRefCreateFromCString(keep_dims_str.as_ptr()),
                    ),
                    mlirBoolAttrGet(context, value.into()),
                ));
            }
            None => (),
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
}

pub fn create_fhelinalg_concat_op(
    context: MlirContext,
    eint_tensors: &[MlirValue],
    axis: Option<i64>,
    result_type: MlirType,
) -> MlirOperation {
    unsafe {
        let mut attrs: Vec<MlirNamedAttribute> = Vec::new();
        match axis {
            Some(value) => {
                let axis_str = CString::new("axis").unwrap();
                attrs.push(mlirNamedAttributeGet(
                    mlirIdentifierGet(context, mlirStringRefCreateFromCString(axis_str.as_ptr())),
                    mlirIntegerAttrGet(mlirIntegerTypeGet(context, 64), value.into()),
                ));
            }
            None => (),
        }
        create_op(
            context,
            "FHELinalg.concat",
            eint_tensors,
            [result_type].as_slice(),
            &attrs,
            false,
        )
    }
}

pub fn create_fhelinalg_conv2d_op(
    context: MlirContext,
    input: MlirValue,
    weight: MlirValue,
    bias: Option<MlirValue>,
    padding: Option<&[i64]>,
    strides: Option<&[i64]>,
    dilations: Option<&[i64]>,
    group: Option<i64>,
    result_type: MlirType,
) -> MlirOperation {
    unsafe {
        let mut operands = Vec::new();
        operands.push(input);
        operands.push(weight);
        match bias {
            Some(value) => operands.push(value),
            None => (),
        }
        let mut attrs = Vec::new();
        match padding {
            Some(value) => {
                let padding_str = CString::new("padding").unwrap();
                attrs.push(mlirNamedAttributeGet(
                    mlirIdentifierGet(
                        context,
                        mlirStringRefCreateFromCString(padding_str.as_ptr()),
                    ),
                    mlirDenseElementsAttrInt64Get(
                        mlirRankedTensorTypeGet(
                            1,
                            [value.len() as i64].as_ptr(),
                            mlirIntegerTypeGet(context, 64),
                            mlirAttributeGetNull(),
                        ),
                        value.len() as isize,
                        value.as_ptr(),
                    ),
                ));
            }
            None => (),
        }
        match strides {
            Some(value) => {
                let strides_str = CString::new("strides").unwrap();
                attrs.push(mlirNamedAttributeGet(
                    mlirIdentifierGet(
                        context,
                        mlirStringRefCreateFromCString(strides_str.as_ptr()),
                    ),
                    mlirDenseElementsAttrInt64Get(
                        mlirRankedTensorTypeGet(
                            1,
                            [value.len() as i64].as_ptr(),
                            mlirIntegerTypeGet(context, 64),
                            mlirAttributeGetNull(),
                        ),
                        value.len() as isize,
                        value.as_ptr(),
                    ),
                ));
            }
            None => (),
        }
        match dilations {
            Some(value) => {
                let dilations_str = CString::new("dilations").unwrap();
                attrs.push(mlirNamedAttributeGet(
                    mlirIdentifierGet(
                        context,
                        mlirStringRefCreateFromCString(dilations_str.as_ptr()),
                    ),
                    mlirDenseElementsAttrInt64Get(
                        mlirRankedTensorTypeGet(
                            1,
                            [value.len() as i64].as_ptr(),
                            mlirIntegerTypeGet(context, 64),
                            mlirAttributeGetNull(),
                        ),
                        value.len() as isize,
                        value.as_ptr(),
                    ),
                ));
            }
            None => (),
        }
        match group {
            Some(value) => {
                let group_str = CString::new("group").unwrap();
                attrs.push(mlirNamedAttributeGet(
                    mlirIdentifierGet(context, mlirStringRefCreateFromCString(group_str.as_ptr())),
                    mlirIntegerAttrGet(mlirIntegerTypeGet(context, 64), value.into()),
                ));
            }
            None => (),
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
}

pub fn create_fhelinalg_transpose_op(
    context: MlirContext,
    eint_tensor: MlirValue,
    axes: Option<&[i64]>,
    result_type: MlirType,
) -> MlirOperation {
    unsafe {
        let mut attrs: Vec<MlirNamedAttribute> = Vec::new();
        match axes {
            Some(value) => {
                let axes_str = CString::new("axes").unwrap();
                let axes_attrs: Vec<MlirAttribute> = value
                    .into_iter()
                    .map(|value| mlirIntegerAttrGet(mlirIntegerTypeGet(context, 64), *value))
                    .collect();
                attrs.push(mlirNamedAttributeGet(
                    mlirIdentifierGet(context, mlirStringRefCreateFromCString(axes_str.as_ptr())),
                    mlirArrayAttrGet(
                        context,
                        value.len().try_into().unwrap(),
                        axes_attrs.as_ptr(),
                    ),
                ));
            }
            None => (),
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
        let input_type = mlirValueGetType(eint_tensor);
        let rank = mlirShapedTypeGetRank(input_type);
        let shape: Vec<i64> = (0i64..rank)
            .map(|dim| mlirShapedTypeGetDimSize(input_type, dim.try_into().unwrap()))
            .collect();
        let results = [mlirRankedTensorTypeGet(
            rank.try_into().unwrap(),
            shape.as_ptr(),
            convert_eint_to_esint_type(context, mlirShapedTypeGetElementType(input_type)).unwrap(),
            mlirAttributeGetNull(),
        )];
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
        let input_type = mlirValueGetType(esint_tensor);
        let rank = mlirShapedTypeGetRank(input_type);
        let shape: Vec<i64> = (0i64..rank)
            .map(|dim| mlirShapedTypeGetDimSize(input_type, dim.try_into().unwrap()))
            .collect();
        let results = [mlirRankedTensorTypeGet(
            rank.try_into().unwrap(),
            shape.as_ptr(),
            convert_esint_to_eint_type(context, mlirShapedTypeGetElementType(input_type)).unwrap(),
            mlirAttributeGetNull(),
        )];
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

    fn get_eint_tensor_type(context: MlirContext, shape: &[i64], width: u32) -> MlirType {
        unsafe {
            let eint_or_error = fheEncryptedIntegerTypeGetChecked(context, width);
            assert!(!eint_or_error.isError);
            let eint = eint_or_error.type_;
            mlirRankedTensorTypeGetChecked(
                mlirLocationUnknownGet(context),
                shape.len().try_into().unwrap(),
                shape.as_ptr(),
                eint,
                mlirAttributeGetNull(),
            )
        }
    }

    fn get_esint_tensor_type(context: MlirContext, shape: &[i64], width: u32) -> MlirType {
        unsafe {
            let eint_or_error = fheEncryptedSignedIntegerTypeGetChecked(context, width);
            assert!(!eint_or_error.isError);
            let eint = eint_or_error.type_;
            mlirRankedTensorTypeGetChecked(
                mlirLocationUnknownGet(context),
                shape.len().try_into().unwrap(),
                shape.as_ptr(),
                eint,
                mlirAttributeGetNull(),
            )
        }
    }

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
    fn test_add_eint_op() {
        unsafe {
            let context = mlirContextCreate();
            mlirRegisterAllDialects(context);
            // register the FHE dialect
            let fhe_handle = mlirGetDialectHandle__fhe__();
            mlirDialectHandleLoadDialect(fhe_handle, context);
            // register the FHELinalg dialect
            let fhelinalg_handle = mlirGetDialectHandle__fhelinalg__();
            mlirDialectHandleLoadDialect(fhelinalg_handle, context);

            // create module for ops
            let location = mlirLocationUnknownGet(context);
            let module = mlirModuleCreateEmpty(location);
            let main_block = mlirModuleGetBody(module);

            // create a 4-bit eint tensor type
            let eint_tensor_type = get_eint_tensor_type(context, &[5, 7], 4);
            // create a zero tensor
            let zero_tensor_op = create_fhe_zero_eint_tensor_op(context, eint_tensor_type);
            mlirBlockAppendOwnedOperation(main_block, zero_tensor_op);
            let eint_tensor_value = mlirOperationGetResult(zero_tensor_op, 0);
            // create add_eint op
            let add_eint_op = create_fhelinalg_add_eint_op(
                context,
                eint_tensor_value,
                eint_tensor_value,
                eint_tensor_type,
            );
            mlirBlockAppendOwnedOperation(main_block, add_eint_op);

            let printed_op = print_mlir_operation_to_string(add_eint_op);
            let expected_op = "%1 = \"FHELinalg.add_eint\"(%0, %0) : (tensor<5x7x!FHE.eint<4>>, tensor<5x7x!FHE.eint<4>>) -> tensor<5x7x!FHE.eint<4>>";
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
            // register the FHELinalg dialect
            let fhelinalg_handle = mlirGetDialectHandle__fhelinalg__();
            mlirDialectHandleLoadDialect(fhelinalg_handle, context);

            // create module for ops
            let location = mlirLocationUnknownGet(context);
            let module = mlirModuleCreateEmpty(location);
            let main_block = mlirModuleGetBody(module);

            // create a 4-bit eint tensor type
            let shape = [73, 1];
            let eint_tensor_type = get_eint_tensor_type(context, &shape, 4);
            // create a zero tensor
            let zero_tensor_op = create_fhe_zero_eint_tensor_op(context, eint_tensor_type);
            mlirBlockAppendOwnedOperation(main_block, zero_tensor_op);
            let eint_tensor_value = mlirOperationGetResult(zero_tensor_op, 0);
            // create constant tensor
            let constant_int_tensor_op = create_constant_tensor_op(context, &shape, &[0], 5);
            mlirBlockAppendOwnedOperation(main_block, constant_int_tensor_op);
            let int_tensor_value = mlirOperationGetResult(constant_int_tensor_op, 0);
            // create add_eint_int op
            let add_eint_int_op = create_fhelinalg_add_eint_int_op(
                context,
                eint_tensor_value,
                int_tensor_value,
                eint_tensor_type,
            );
            mlirBlockAppendOwnedOperation(main_block, add_eint_int_op);

            let printed_op = print_mlir_operation_to_string(add_eint_int_op);
            let expected_op = "%1 = \"FHELinalg.add_eint_int\"(%0, %cst) : (tensor<73x1x!FHE.eint<4>>, tensor<73x1xi5>) -> tensor<73x1x!FHE.eint<4>>";
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
            // register the FHELinalg dialect
            let fhelinalg_handle = mlirGetDialectHandle__fhelinalg__();
            mlirDialectHandleLoadDialect(fhelinalg_handle, context);

            // create module for ops
            let location = mlirLocationUnknownGet(context);
            let module = mlirModuleCreateEmpty(location);
            let main_block = mlirModuleGetBody(module);

            // create a 4-bit eint tensor type
            let eint_tensor_type = get_eint_tensor_type(context, &[5, 7], 4);
            // create a zero tensor
            let zero_tensor_op = create_fhe_zero_eint_tensor_op(context, eint_tensor_type);
            mlirBlockAppendOwnedOperation(main_block, zero_tensor_op);
            let eint_tensor_value = mlirOperationGetResult(zero_tensor_op, 0);
            // create sub_eint op
            let sub_eint_op = create_fhelinalg_sub_eint_op(
                context,
                eint_tensor_value,
                eint_tensor_value,
                eint_tensor_type,
            );
            mlirBlockAppendOwnedOperation(main_block, sub_eint_op);

            let printed_op = print_mlir_operation_to_string(sub_eint_op);
            let expected_op = "%1 = \"FHELinalg.sub_eint\"(%0, %0) : (tensor<5x7x!FHE.eint<4>>, tensor<5x7x!FHE.eint<4>>) -> tensor<5x7x!FHE.eint<4>>";
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
            // register the FHELinalg dialect
            let fhelinalg_handle = mlirGetDialectHandle__fhelinalg__();
            mlirDialectHandleLoadDialect(fhelinalg_handle, context);

            // create module for ops
            let location = mlirLocationUnknownGet(context);
            let module = mlirModuleCreateEmpty(location);
            let main_block = mlirModuleGetBody(module);

            // create a 4-bit eint tensor type
            let shape = [2, 4, 6, 9, 13, 100];
            let eint_tensor_type = get_eint_tensor_type(context, &shape, 4);
            // create a zero tensor
            let zero_tensor_op = create_fhe_zero_eint_tensor_op(context, eint_tensor_type);
            mlirBlockAppendOwnedOperation(main_block, zero_tensor_op);
            let eint_tensor_value = mlirOperationGetResult(zero_tensor_op, 0);
            // create constant tensor
            let constant_int_tensor_op = create_constant_tensor_op(context, &shape, &[0], 5);
            mlirBlockAppendOwnedOperation(main_block, constant_int_tensor_op);
            let int_tensor_value = mlirOperationGetResult(constant_int_tensor_op, 0);
            // create sub_eint_int op
            let sub_eint_int_op = create_fhelinalg_sub_eint_int_op(
                context,
                eint_tensor_value,
                int_tensor_value,
                eint_tensor_type,
            );
            mlirBlockAppendOwnedOperation(main_block, sub_eint_int_op);

            let printed_op = print_mlir_operation_to_string(sub_eint_int_op);
            let expected_op = "%1 = \"FHELinalg.sub_eint_int\"(%0, %cst) : (tensor<2x4x6x9x13x100x!FHE.eint<4>>, tensor<2x4x6x9x13x100xi5>) \
-> tensor<2x4x6x9x13x100x!FHE.eint<4>>";
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
            // register the FHELinalg dialect
            let fhelinalg_handle = mlirGetDialectHandle__fhelinalg__();
            mlirDialectHandleLoadDialect(fhelinalg_handle, context);

            // create module for ops
            let location = mlirLocationUnknownGet(context);
            let module = mlirModuleCreateEmpty(location);
            let main_block = mlirModuleGetBody(module);

            // create a 4-bit eint tensor type
            let shape = [1];
            let eint_tensor_type = get_eint_tensor_type(context, &shape, 4);
            // create a zero tensor
            let zero_tensor_op = create_fhe_zero_eint_tensor_op(context, eint_tensor_type);
            mlirBlockAppendOwnedOperation(main_block, zero_tensor_op);
            let eint_tensor_value = mlirOperationGetResult(zero_tensor_op, 0);
            // create constant tensor
            let constant_int_tensor_op = create_constant_tensor_op(context, &shape, &[0], 5);
            mlirBlockAppendOwnedOperation(main_block, constant_int_tensor_op);
            let int_tensor_value = mlirOperationGetResult(constant_int_tensor_op, 0);
            // create sub_int_eint op
            let sub_int_eint_op = create_fhelinalg_sub_int_eint_op(
                context,
                eint_tensor_value,
                int_tensor_value,
                eint_tensor_type,
            );
            mlirBlockAppendOwnedOperation(main_block, sub_int_eint_op);

            let printed_op = print_mlir_operation_to_string(sub_int_eint_op);
            let expected_op = "%2 = \"FHELinalg.sub_int_eint\"(%0, %1) : (tensor<1x!FHE.eint<4>>, tensor<1xi5>) -> tensor<1x!FHE.eint<4>>";
            assert_eq!(printed_op, expected_op);
        }
    }

    #[test]
    fn test_neg_eint_op() {
        unsafe {
            let context = mlirContextCreate();
            mlirRegisterAllDialects(context);
            // register the FHE dialect
            let fhe_handle = mlirGetDialectHandle__fhe__();
            mlirDialectHandleLoadDialect(fhe_handle, context);
            // register the FHELinalg dialect
            let fhelinalg_handle = mlirGetDialectHandle__fhelinalg__();
            mlirDialectHandleLoadDialect(fhelinalg_handle, context);

            // create module for ops
            let location = mlirLocationUnknownGet(context);
            let module = mlirModuleCreateEmpty(location);
            let main_block = mlirModuleGetBody(module);

            // create a 4-bit eint tensor type
            let shape = [16];
            let eint_tensor_type = get_eint_tensor_type(context, &shape, 4);
            // create a zero tensor
            let zero_tensor_op = create_fhe_zero_eint_tensor_op(context, eint_tensor_type);
            mlirBlockAppendOwnedOperation(main_block, zero_tensor_op);
            let eint_tensor_value = mlirOperationGetResult(zero_tensor_op, 0);
            // create neg_eint op
            let neg_eint_op = create_fhelinalg_negate_eint_op(context, eint_tensor_value);
            mlirBlockAppendOwnedOperation(main_block, neg_eint_op);

            let printed_op = print_mlir_operation_to_string(neg_eint_op);
            let expected_op = "%1 = \"FHELinalg.neg_eint\"(%0) : (tensor<16x!FHE.eint<4>>) -> tensor<16x!FHE.eint<4>>";
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
            // register the FHELinalg dialect
            let fhelinalg_handle = mlirGetDialectHandle__fhelinalg__();
            mlirDialectHandleLoadDialect(fhelinalg_handle, context);

            // create module for ops
            let location = mlirLocationUnknownGet(context);
            let module = mlirModuleCreateEmpty(location);
            let main_block = mlirModuleGetBody(module);

            // create a 4-bit eint tensor type
            let shape = [100];
            let eint_tensor_type = get_eint_tensor_type(context, &shape, 4);
            // create a zero tensor
            let zero_tensor_op = create_fhe_zero_eint_tensor_op(context, eint_tensor_type);
            mlirBlockAppendOwnedOperation(main_block, zero_tensor_op);
            let eint_tensor_value = mlirOperationGetResult(zero_tensor_op, 0);
            // create constant tensor
            let constant_int_tensor_op = create_constant_tensor_op(context, &shape, &[0], 5);
            mlirBlockAppendOwnedOperation(main_block, constant_int_tensor_op);
            let int_tensor_value = mlirOperationGetResult(constant_int_tensor_op, 0);
            // create mul_eint_int op
            let mul_eint_int_op = create_fhelinalg_mul_eint_int_op(
                context,
                eint_tensor_value,
                int_tensor_value,
                eint_tensor_type,
            );
            mlirBlockAppendOwnedOperation(main_block, mul_eint_int_op);

            let printed_op = print_mlir_operation_to_string(mul_eint_int_op);
            let expected_op = "%1 = \"FHELinalg.mul_eint_int\"(%0, %cst) : (tensor<100x!FHE.eint<4>>, tensor<100xi5>) \
-> tensor<100x!FHE.eint<4>>";
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
            // register the FHELinalg dialect
            let fhelinalg_handle = mlirGetDialectHandle__fhelinalg__();
            mlirDialectHandleLoadDialect(fhelinalg_handle, context);

            // create module for ops
            let location = mlirLocationUnknownGet(context);
            let module = mlirModuleCreateEmpty(location);
            let main_block = mlirModuleGetBody(module);

            // create a 4-bit eint tensor type
            let shape_tensor = [4, 4, 4];
            let eint_tensor_type = get_eint_tensor_type(context, &shape_tensor, 4);
            // create a zero tensor
            let zero_tensor_op = create_fhe_zero_eint_tensor_op(context, eint_tensor_type);
            mlirBlockAppendOwnedOperation(main_block, zero_tensor_op);
            let eint_tensor_value = mlirOperationGetResult(zero_tensor_op, 0);
            // create constant tensor
            let constant_int_tensor_op = create_constant_tensor_op(context, &[16], &[0], 64);
            mlirBlockAppendOwnedOperation(main_block, constant_int_tensor_op);
            let lut = mlirOperationGetResult(constant_int_tensor_op, 0);
            // create lut op
            let lut_op =
                create_fhelinalg_apply_lut_op(context, eint_tensor_value, lut, eint_tensor_type);
            mlirBlockAppendOwnedOperation(main_block, lut_op);

            let printed_op = print_mlir_operation_to_string(lut_op);
            let expected_op = "%1 = \"FHELinalg.apply_lookup_table\"(%0, %cst) : (tensor<4x4x4x!FHE.eint<4>>, tensor<16xi64>) \
-> tensor<4x4x4x!FHE.eint<4>>";
            assert_eq!(printed_op, expected_op);
        }
    }

    #[test]
    fn test_apply_multi_lut_op() {
        unsafe {
            let context = mlirContextCreate();
            mlirRegisterAllDialects(context);
            // register the FHE dialect
            let fhe_handle = mlirGetDialectHandle__fhe__();
            mlirDialectHandleLoadDialect(fhe_handle, context);
            // register the FHELinalg dialect
            let fhelinalg_handle = mlirGetDialectHandle__fhelinalg__();
            mlirDialectHandleLoadDialect(fhelinalg_handle, context);

            // create module for ops
            let location = mlirLocationUnknownGet(context);
            let module = mlirModuleCreateEmpty(location);
            let main_block = mlirModuleGetBody(module);

            // create a 4-bit eint tensor type
            let shape_tensor = [4, 4, 4];
            let eint_tensor_type = get_eint_tensor_type(context, &shape_tensor, 4);
            // create a zero tensor
            let zero_tensor_op = create_fhe_zero_eint_tensor_op(context, eint_tensor_type);
            mlirBlockAppendOwnedOperation(main_block, zero_tensor_op);
            let eint_tensor_value = mlirOperationGetResult(zero_tensor_op, 0);
            // create constant tensor
            let constant_int_tensor_op =
                create_constant_tensor_op(context, &[4, 4, 4, 16], &[0], 64);
            mlirBlockAppendOwnedOperation(main_block, constant_int_tensor_op);
            let lut = mlirOperationGetResult(constant_int_tensor_op, 0);
            // create lut op
            let lut_op = create_fhelinalg_apply_multi_lut_op(
                context,
                eint_tensor_value,
                lut,
                eint_tensor_type,
            );
            mlirBlockAppendOwnedOperation(main_block, lut_op);

            let printed_op = print_mlir_operation_to_string(lut_op);
            let expected_op = "%1 = \"FHELinalg.apply_multi_lookup_table\"(%0, %cst) : (tensor<4x4x4x!FHE.eint<4>>, tensor<4x4x4x16xi64>) \
-> tensor<4x4x4x!FHE.eint<4>>";
            assert_eq!(printed_op, expected_op);
        }
    }

    #[test]
    fn test_apply_mapped_lut_op() {
        unsafe {
            let context = mlirContextCreate();
            mlirRegisterAllDialects(context);
            // register the FHE dialect
            let fhe_handle = mlirGetDialectHandle__fhe__();
            mlirDialectHandleLoadDialect(fhe_handle, context);
            // register the FHELinalg dialect
            let fhelinalg_handle = mlirGetDialectHandle__fhelinalg__();
            mlirDialectHandleLoadDialect(fhelinalg_handle, context);

            // create module for ops
            let location = mlirLocationUnknownGet(context);
            let module = mlirModuleCreateEmpty(location);
            let main_block = mlirModuleGetBody(module);

            // create a 4-bit eint tensor type
            let shape_tensor = [4, 4, 4];
            let eint_tensor_type = get_eint_tensor_type(context, &shape_tensor, 4);
            // create a zero tensor
            let zero_tensor_op = create_fhe_zero_eint_tensor_op(context, eint_tensor_type);
            mlirBlockAppendOwnedOperation(main_block, zero_tensor_op);
            let eint_tensor_value = mlirOperationGetResult(zero_tensor_op, 0);
            // create constant tensor
            let constant_int_tensor_op = create_constant_tensor_op(context, &[5, 16], &[0], 64);
            mlirBlockAppendOwnedOperation(main_block, constant_int_tensor_op);
            let lut = mlirOperationGetResult(constant_int_tensor_op, 0);
            // create map tensor
            let constant_int_map_tensor_op =
                create_constant_tensor_op(context, &[4, 4, 4], &[0], 64);
            mlirBlockAppendOwnedOperation(main_block, constant_int_map_tensor_op);
            let map = mlirOperationGetResult(constant_int_map_tensor_op, 0);
            // create lut op
            let lut_op = create_fhelinalg_apply_mapped_lut_op(
                context,
                eint_tensor_value,
                lut,
                map,
                eint_tensor_type,
            );
            mlirBlockAppendOwnedOperation(main_block, lut_op);

            let printed_op = print_mlir_operation_to_string(lut_op);
            let expected_op = "%3 = \"FHELinalg.apply_mapped_lookup_table\"(%0, %1, %2) : (tensor<4x4x4x!FHE.eint<4>>, tensor<5x16xi64>, tensor<4x4x4xi64>) \
-> tensor<4x4x4x!FHE.eint<4>>";
            assert_eq!(printed_op, expected_op);
        }
    }

    #[test]
    fn test_dot_eint_int_op() {
        unsafe {
            let context = mlirContextCreate();
            mlirRegisterAllDialects(context);
            // register the FHE dialect
            let fhe_handle = mlirGetDialectHandle__fhe__();
            mlirDialectHandleLoadDialect(fhe_handle, context);
            // register the FHELinalg dialect
            let fhelinalg_handle = mlirGetDialectHandle__fhelinalg__();
            mlirDialectHandleLoadDialect(fhelinalg_handle, context);

            // create module for ops
            let location = mlirLocationUnknownGet(context);
            let module = mlirModuleCreateEmpty(location);
            let main_block = mlirModuleGetBody(module);

            // create a 4-bit eint tensor type
            let shape = [100];
            let eint_tensor_type = get_eint_tensor_type(context, &shape, 4);
            // create a zero tensor
            let zero_tensor_op = create_fhe_zero_eint_tensor_op(context, eint_tensor_type);
            mlirBlockAppendOwnedOperation(main_block, zero_tensor_op);
            let eint_tensor_value = mlirOperationGetResult(zero_tensor_op, 0);
            // create constant tensor
            let constant_int_tensor_op = create_constant_tensor_op(context, &shape, &[0], 5);
            mlirBlockAppendOwnedOperation(main_block, constant_int_tensor_op);
            let int_tensor_value = mlirOperationGetResult(constant_int_tensor_op, 0);
            // create dot_eint_int op
            let dot_eint_int_op = create_fhelinalg_dot_eint_int_op(
                context,
                eint_tensor_value,
                int_tensor_value,
                eint_tensor_type,
            );
            mlirBlockAppendOwnedOperation(main_block, dot_eint_int_op);

            let printed_op = print_mlir_operation_to_string(dot_eint_int_op);
            let expected_op = "%2 = \"FHELinalg.dot_eint_int\"(%0, %1) : (tensor<100x!FHE.eint<4>>, tensor<100xi5>) -> tensor<100x!FHE.eint<4>>";
            assert_eq!(printed_op, expected_op);
        }
    }

    #[test]
    fn test_matmul_eint_int_op() {
        unsafe {
            let context = mlirContextCreate();
            mlirRegisterAllDialects(context);
            // register the FHE dialect
            let fhe_handle = mlirGetDialectHandle__fhe__();
            mlirDialectHandleLoadDialect(fhe_handle, context);
            // register the FHELinalg dialect
            let fhelinalg_handle = mlirGetDialectHandle__fhelinalg__();
            mlirDialectHandleLoadDialect(fhelinalg_handle, context);

            // create module for ops
            let location = mlirLocationUnknownGet(context);
            let module = mlirModuleCreateEmpty(location);
            let main_block = mlirModuleGetBody(module);

            // create a 4-bit eint tensor type
            let shape = [5, 5];
            let eint_tensor_type = get_eint_tensor_type(context, &shape, 4);
            // create a zero tensor
            let zero_tensor_op = create_fhe_zero_eint_tensor_op(context, eint_tensor_type);
            mlirBlockAppendOwnedOperation(main_block, zero_tensor_op);
            let eint_tensor_value = mlirOperationGetResult(zero_tensor_op, 0);
            // create constant tensor
            let constant_int_tensor_op = create_constant_tensor_op(context, &shape, &[0], 5);
            mlirBlockAppendOwnedOperation(main_block, constant_int_tensor_op);
            let int_tensor_value = mlirOperationGetResult(constant_int_tensor_op, 0);
            // create matmul_eint_int op
            let matmul_eint_int_op = create_fhelinalg_matmul_eint_int_op(
                context,
                eint_tensor_value,
                int_tensor_value,
                eint_tensor_type,
            );
            mlirBlockAppendOwnedOperation(main_block, matmul_eint_int_op);

            let printed_op = print_mlir_operation_to_string(matmul_eint_int_op);
            let expected_op = "%1 = \"FHELinalg.matmul_eint_int\"(%0, %cst) : (tensor<5x5x!FHE.eint<4>>, tensor<5x5xi5>) -> tensor<5x5x!FHE.eint<4>>";
            assert_eq!(printed_op, expected_op);
        }
    }

    #[test]
    fn test_matmul_int_eint_op() {
        unsafe {
            let context = mlirContextCreate();
            mlirRegisterAllDialects(context);
            // register the FHE dialect
            let fhe_handle = mlirGetDialectHandle__fhe__();
            mlirDialectHandleLoadDialect(fhe_handle, context);
            // register the FHELinalg dialect
            let fhelinalg_handle = mlirGetDialectHandle__fhelinalg__();
            mlirDialectHandleLoadDialect(fhelinalg_handle, context);

            // create module for ops
            let location = mlirLocationUnknownGet(context);
            let module = mlirModuleCreateEmpty(location);
            let main_block = mlirModuleGetBody(module);

            // create a 4-bit eint tensor type
            let shape = [5, 5];
            let eint_tensor_type = get_eint_tensor_type(context, &shape, 4);
            // create a zero tensor
            let zero_tensor_op = create_fhe_zero_eint_tensor_op(context, eint_tensor_type);
            mlirBlockAppendOwnedOperation(main_block, zero_tensor_op);
            let eint_tensor_value = mlirOperationGetResult(zero_tensor_op, 0);
            // create constant tensor
            let constant_int_tensor_op = create_constant_tensor_op(context, &shape, &[0], 5);
            mlirBlockAppendOwnedOperation(main_block, constant_int_tensor_op);
            let int_tensor_value = mlirOperationGetResult(constant_int_tensor_op, 0);
            // create matmul_int_eint op
            let matmul_int_eint_op = create_fhelinalg_matmul_int_eint_op(
                context,
                int_tensor_value,
                eint_tensor_value,
                eint_tensor_type,
            );
            mlirBlockAppendOwnedOperation(main_block, matmul_int_eint_op);

            let printed_op = print_mlir_operation_to_string(matmul_int_eint_op);
            let expected_op = "%1 = \"FHELinalg.matmul_int_eint\"(%cst, %0) : (tensor<5x5xi5>, tensor<5x5x!FHE.eint<4>>) -> tensor<5x5x!FHE.eint<4>>";
            assert_eq!(printed_op, expected_op);
        }
    }

    #[test]
    fn test_sum_eint_op() {
        unsafe {
            let context = mlirContextCreate();
            mlirRegisterAllDialects(context);
            // register the FHE dialect
            let fhe_handle = mlirGetDialectHandle__fhe__();
            mlirDialectHandleLoadDialect(fhe_handle, context);
            // register the FHELinalg dialect
            let fhelinalg_handle = mlirGetDialectHandle__fhelinalg__();
            mlirDialectHandleLoadDialect(fhelinalg_handle, context);

            // create module for ops
            let location = mlirLocationUnknownGet(context);
            let module = mlirModuleCreateEmpty(location);
            let main_block = mlirModuleGetBody(module);

            // create a 4-bit eint tensor type
            let shape = [5, 5];
            let eint_tensor_type = get_eint_tensor_type(context, &shape, 4);
            // create a zero tensor
            let zero_tensor_op = create_fhe_zero_eint_tensor_op(context, eint_tensor_type);
            mlirBlockAppendOwnedOperation(main_block, zero_tensor_op);
            let eint_tensor_value = mlirOperationGetResult(zero_tensor_op, 0);
            // create sum op
            let sum_eint_op = create_fhelinalg_sum_op(
                context,
                eint_tensor_value,
                Some(&[1]),
                Some(false),
                get_eint_tensor_type(context, &[5], 4),
            );
            mlirBlockAppendOwnedOperation(main_block, sum_eint_op);

            let printed_op = print_mlir_operation_to_string(sum_eint_op);
            let expected_op = "%1 = \"FHELinalg.sum\"(%0) {axes = [1], keep_dims = false} : (tensor<5x5x!FHE.eint<4>>) -> tensor<5x!FHE.eint<4>>";
            assert_eq!(printed_op, expected_op);
        }
    }

    #[test]
    fn test_concat_eint_op() {
        unsafe {
            let context = mlirContextCreate();
            mlirRegisterAllDialects(context);
            // register the FHE dialect
            let fhe_handle = mlirGetDialectHandle__fhe__();
            mlirDialectHandleLoadDialect(fhe_handle, context);
            // register the FHELinalg dialect
            let fhelinalg_handle = mlirGetDialectHandle__fhelinalg__();
            mlirDialectHandleLoadDialect(fhelinalg_handle, context);

            // create module for ops
            let location = mlirLocationUnknownGet(context);
            let module = mlirModuleCreateEmpty(location);
            let main_block = mlirModuleGetBody(module);

            // create a 4-bit eint tensor type
            let shape = [3, 3];
            let eint_tensor_type = get_eint_tensor_type(context, &shape, 4);
            // create a zero tensor
            let zero_tensor_op = create_fhe_zero_eint_tensor_op(context, eint_tensor_type);
            mlirBlockAppendOwnedOperation(main_block, zero_tensor_op);
            let eint_tensor_value = mlirOperationGetResult(zero_tensor_op, 0);
            // create concat op
            let concat_eint_op = create_fhelinalg_concat_op(
                context,
                &[eint_tensor_value, eint_tensor_value],
                Some(0),
                get_eint_tensor_type(context, &[6, 3], 4),
            );
            mlirBlockAppendOwnedOperation(main_block, concat_eint_op);

            let printed_op = print_mlir_operation_to_string(concat_eint_op);
            let expected_op = "%1 = \"FHELinalg.concat\"(%0, %0) {axis = 0 : i64} : (tensor<3x3x!FHE.eint<4>>, tensor<3x3x!FHE.eint<4>>) -> \
tensor<6x3x!FHE.eint<4>>";
            assert_eq!(printed_op, expected_op);
        }
    }

    #[test]
    fn test_conv2d_eint_int_op() {
        unsafe {
            let context = mlirContextCreate();
            mlirRegisterAllDialects(context);
            // register the FHE dialect
            let fhe_handle = mlirGetDialectHandle__fhe__();
            mlirDialectHandleLoadDialect(fhe_handle, context);
            // register the FHELinalg dialect
            let fhelinalg_handle = mlirGetDialectHandle__fhelinalg__();
            mlirDialectHandleLoadDialect(fhelinalg_handle, context);

            // create module for ops
            let location = mlirLocationUnknownGet(context);
            let module = mlirModuleCreateEmpty(location);
            let main_block = mlirModuleGetBody(module);

            // create a 4-bit eint tensor type
            let eint_tensor_type = get_eint_tensor_type(context, &[100, 3, 28, 28], 4);
            // create a zero tensor as input
            let zero_tensor_op = create_fhe_zero_eint_tensor_op(context, eint_tensor_type);
            mlirBlockAppendOwnedOperation(main_block, zero_tensor_op);
            let input = mlirOperationGetResult(zero_tensor_op, 0);
            // create constant weight tensor
            let constant_int_tensor_op =
                create_constant_tensor_op(context, &[4, 3, 14, 14], &[0], 5);
            mlirBlockAppendOwnedOperation(main_block, constant_int_tensor_op);
            let weight = mlirOperationGetResult(constant_int_tensor_op, 0);
            // create constant bias tensor
            let constant_int_tensor_op = create_constant_tensor_op(context, &[4], &[0], 5);
            mlirBlockAppendOwnedOperation(main_block, constant_int_tensor_op);
            let bias = mlirOperationGetResult(constant_int_tensor_op, 0);
            // create matmul_eint_int op
            let conv2d_op = create_fhelinalg_conv2d_op(
                context,
                input,
                weight,
                Some(bias),
                Some(&[0, 0, 0, 0]),
                Some(&[1, 1]),
                Some(&[1, 1]),
                Some(1),
                get_eint_tensor_type(context, &[100, 4, 15, 15], 4),
            );
            mlirBlockAppendOwnedOperation(main_block, conv2d_op);

            let printed_op = print_mlir_operation_to_string(conv2d_op);
            let expected_op = "%1 = \"FHELinalg.conv2d\"(%0, %cst, %cst_0) {dilations = dense<1> : tensor<2xi64>, group = 1 : i64, \
padding = dense<0> : tensor<4xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<100x3x28x28x!FHE.eint<4>>, tensor<4x3x14x14xi5>, tensor<4xi5>) \
-> tensor<100x4x15x15x!FHE.eint<4>>";
            assert_eq!(printed_op, expected_op);
        }
    }

    #[test]
    fn test_transpose_eint_op() {
        unsafe {
            let context = mlirContextCreate();
            mlirRegisterAllDialects(context);
            // register the FHE dialect
            let fhe_handle = mlirGetDialectHandle__fhe__();
            mlirDialectHandleLoadDialect(fhe_handle, context);
            // register the FHELinalg dialect
            let fhelinalg_handle = mlirGetDialectHandle__fhelinalg__();
            mlirDialectHandleLoadDialect(fhelinalg_handle, context);

            // create module for ops
            let location = mlirLocationUnknownGet(context);
            let module = mlirModuleCreateEmpty(location);
            let main_block = mlirModuleGetBody(module);

            // create a 4-bit eint tensor type
            let shape = [2, 3, 4, 5];
            let eint_tensor_type = get_eint_tensor_type(context, &shape, 4);
            // create a zero tensor
            let zero_tensor_op = create_fhe_zero_eint_tensor_op(context, eint_tensor_type);
            mlirBlockAppendOwnedOperation(main_block, zero_tensor_op);
            let eint_tensor_value = mlirOperationGetResult(zero_tensor_op, 0);
            // create transpose op
            let transpose_eint_op = create_fhelinalg_transpose_op(
                context,
                eint_tensor_value,
                Some(&[1, 3, 0, 2]),
                get_eint_tensor_type(context, &[3, 5, 2, 4], 4),
            );
            mlirBlockAppendOwnedOperation(main_block, transpose_eint_op);

            let printed_op = print_mlir_operation_to_string(transpose_eint_op);
            let expected_op = "%1 = \"FHELinalg.transpose\"(%0) {axes = [1, 3, 0, 2]} : (tensor<2x3x4x5x!FHE.eint<4>>) -> tensor<3x5x2x4x!FHE.eint<4>>";
            assert_eq!(printed_op, expected_op);
        }
    }

    #[test]
    fn test_from_element_op() {
        unsafe {
            let context = mlirContextCreate();
            mlirRegisterAllDialects(context);
            // register the FHE dialect
            let fhe_handle = mlirGetDialectHandle__fhe__();
            mlirDialectHandleLoadDialect(fhe_handle, context);
            // register the FHELinalg dialect
            let fhelinalg_handle = mlirGetDialectHandle__fhelinalg__();
            mlirDialectHandleLoadDialect(fhelinalg_handle, context);

            // create module for ops
            let location = mlirLocationUnknownGet(context);
            let module = mlirModuleCreateEmpty(location);
            let main_block = mlirModuleGetBody(module);

            // create a 2-bit eint type
            let eint_or_error = fheEncryptedIntegerTypeGetChecked(context, 2);
            assert!(!eint_or_error.isError);
            let eint2_type = eint_or_error.type_;
            // create a zero eint
            let zero_op = create_fhe_zero_eint_tensor_op(context, eint2_type);
            mlirBlockAppendOwnedOperation(main_block, zero_op);
            let value = mlirOperationGetResult(zero_op, 0);
            // create from element op
            let from_element_op = create_fhelinalg_from_element_op(context, value);
            mlirBlockAppendOwnedOperation(main_block, from_element_op);

            let printed_op = print_mlir_operation_to_string(from_element_op);
            let expected_op =
                "%1 = \"FHELinalg.from_element\"(%0) : (!FHE.eint<2>) -> tensor<1x!FHE.eint<2>>";
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
            // register the FHELinalg dialect
            let fhelinalg_handle = mlirGetDialectHandle__fhelinalg__();
            mlirDialectHandleLoadDialect(fhelinalg_handle, context);

            // create module for ops
            let location = mlirLocationUnknownGet(context);
            let module = mlirModuleCreateEmpty(location);
            let main_block = mlirModuleGetBody(module);

            // create a 4-bit eint tensor type
            let shape = [2, 3, 4, 5];
            let eint_tensor_type = get_eint_tensor_type(context, &shape, 4);
            // create a zero tensor
            let zero_tensor_op = create_fhe_zero_eint_tensor_op(context, eint_tensor_type);
            mlirBlockAppendOwnedOperation(main_block, zero_tensor_op);
            let eint_tensor_value = mlirOperationGetResult(zero_tensor_op, 0);
            // create to_signed op
            let to_signed_op = create_fhelinalg_to_signed_op(context, eint_tensor_value);
            mlirBlockAppendOwnedOperation(main_block, to_signed_op);

            let printed_op = print_mlir_operation_to_string(to_signed_op);
            let expected_op = "%1 = \"FHELinalg.to_signed\"(%0) : (tensor<2x3x4x5x!FHE.eint<4>>) -> tensor<2x3x4x5x!FHE.esint<4>>";
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
            // register the FHELinalg dialect
            let fhelinalg_handle = mlirGetDialectHandle__fhelinalg__();
            mlirDialectHandleLoadDialect(fhelinalg_handle, context);

            // create module for ops
            let location = mlirLocationUnknownGet(context);
            let module = mlirModuleCreateEmpty(location);
            let main_block = mlirModuleGetBody(module);

            // create a 4-bit eint tensor type
            let shape = [2, 3, 4, 5];
            let eint_tensor_type = get_esint_tensor_type(context, &shape, 4);
            // create a zero tensor
            let zero_tensor_op = create_fhe_zero_eint_tensor_op(context, eint_tensor_type);
            mlirBlockAppendOwnedOperation(main_block, zero_tensor_op);
            let eint_tensor_value = mlirOperationGetResult(zero_tensor_op, 0);
            // create to_unsigned op
            let to_unsigned_op = create_fhelinalg_to_unsigned_op(context, eint_tensor_value);
            mlirBlockAppendOwnedOperation(main_block, to_unsigned_op);

            let printed_op = print_mlir_operation_to_string(to_unsigned_op);
            let expected_op = "%1 = \"FHELinalg.to_unsigned\"(%0) : (tensor<2x3x4x5x!FHE.esint<4>>) -> tensor<2x3x4x5x!FHE.eint<4>>";
            assert_eq!(printed_op, expected_op);
        }
    }
}
