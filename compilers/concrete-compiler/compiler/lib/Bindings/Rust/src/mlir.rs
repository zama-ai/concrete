//! MLIR module

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

pub mod ffi {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

use ffi::*;
use std::ffi::CString;
use std::ops::AddAssign;

pub(crate) unsafe extern "C" fn mlir_rust_string_receiver_callback(
    mlirStrRef: MlirStringRef,
    user_data: *mut ::std::os::raw::c_void,
) {
    let rust_string = &mut *(user_data as *mut String);
    let slc = std::slice::from_raw_parts(mlirStrRef.data as *const u8, mlirStrRef.length as usize);
    rust_string.add_assign(&String::from_utf8_lossy(slc));
}

pub fn print_mlir_operation_to_string(op: MlirOperation) -> String {
    let mut rust_string = String::default();
    let receiver_ptr = (&mut rust_string) as *mut String as *mut ::std::os::raw::c_void;

    unsafe {
        mlirOperationPrint(op, Some(mlir_rust_string_receiver_callback), receiver_ptr);
    }

    rust_string
}

pub fn print_mlir_type_to_string(mlir_type: MlirType) -> String {
    let mut rust_string = String::default();
    let receiver_ptr = (&mut rust_string) as *mut String as *mut ::std::os::raw::c_void;

    unsafe {
        mlirTypePrint(
            mlir_type,
            Some(mlir_rust_string_receiver_callback),
            receiver_ptr,
        );
    }

    rust_string
}

/// Returns a function operation with a region that contains a block.
///
/// The function would be defined using the provided input and output types. The main block of the
/// function can be later fetched, from which we can get function arguments, and it will be where
/// we append operations.
///
/// # Examples
/// ```
/// use concrete_compiler::mlir::*;
/// use concrete_compiler::mlir::ffi::*;
/// unsafe{
///     let context = mlirContextCreate();
///     register_all_dialects(context);
///
///     // input/output types
///     let func_input_types = [
///         mlirIntegerTypeGet(context, 64),
///         mlirIntegerTypeGet(context, 64),
///     ];
///     let func_output_types = [mlirIntegerTypeGet(context, 64)];
///
///     let func_op = create_func_with_block(
///         context,
///         "test",
///         func_input_types.as_slice(),
///         func_output_types.as_slice(),
///     );
///
///     // we can fetch the main block of the function from the function region
///     let func_block = mlirRegionGetFirstBlock(mlirOperationGetFirstRegion(func_op));
///     // we can get arguments to later be used as operands to other operations
///     let func_args = [
///         mlirBlockGetArgument(func_block, 0),
///         mlirBlockGetArgument(func_block, 1),
///     ];
///     // to add an operation to the function, we will append it to the main block
///     let addi_op = create_addi_op(context, func_args[0], func_args[1]);
///     mlirBlockAppendOwnedOperation(func_block, addi_op);
/// }
/// ```
///
pub fn create_func_with_block(
    context: MlirContext,
    func_name: &str,
    func_input_types: &[MlirType],
    func_output_types: &[MlirType],
) -> MlirOperation {
    unsafe {
        // create the main block of the function
        let locations = (0..func_input_types.len())
            .into_iter()
            .map(|_| mlirLocationUnknownGet(context))
            .collect::<Vec<_>>();
        let func_block = mlirBlockCreate(
            func_input_types.len().try_into().unwrap(),
            func_input_types.as_ptr(),
            locations.as_ptr(),
        );

        // create region to hold the previously created block
        let func_region = mlirRegionCreate();
        mlirRegionAppendOwnedBlock(func_region, func_block);

        // create function to hold the previously created region
        let location = mlirLocationUnknownGet(context);
        let func_str = CString::new("func.func").unwrap();
        let mut func_op_state =
            mlirOperationStateGet(mlirStringRefCreateFromCString(func_str.as_ptr()), location);
        mlirOperationStateAddOwnedRegions(&mut func_op_state, 1, [func_region].as_ptr());
        // set function attributes
        let func_type_str = CString::new("function_type").unwrap();
        let sym_name_str = CString::new("sym_name").unwrap();
        let func_name_str = CString::new(func_name).unwrap();
        let func_type_attr = mlirTypeAttrGet(mlirFunctionTypeGet(
            context,
            func_input_types.len().try_into().unwrap(),
            func_input_types.as_ptr(),
            func_output_types.len().try_into().unwrap(),
            func_output_types.as_ptr(),
        ));
        let sym_name_attr = mlirStringAttrGet(
            context,
            mlirStringRefCreateFromCString(func_name_str.as_ptr()),
        );
        mlirOperationStateAddAttributes(
            &mut func_op_state,
            2,
            [
                // func type
                mlirNamedAttributeGet(
                    mlirIdentifierGet(
                        context,
                        mlirStringRefCreateFromCString(func_type_str.as_ptr()),
                    ),
                    func_type_attr,
                ),
                // func name
                mlirNamedAttributeGet(
                    mlirIdentifierGet(
                        context,
                        mlirStringRefCreateFromCString(sym_name_str.as_ptr()),
                    ),
                    sym_name_attr,
                ),
            ]
            .as_ptr(),
        );
        let func_op = mlirOperationCreate(&mut func_op_state);

        func_op
    }
}

/// Generic function to create an MLIR operation.
///
/// Create an MLIR operation based on its mnemonic (e.g. addi), it's operands, result types, and
/// attributes. Result types can be inferred automatically if the operation itself supports that.
pub fn create_op(
    context: MlirContext,
    mnemonic: &str,
    operands: &[MlirValue],
    results: &[MlirType],
    attrs: &[MlirNamedAttribute],
    auto_result_type_inference: bool,
) -> MlirOperation {
    let op_mnemonic = CString::new(mnemonic).unwrap();
    unsafe {
        let location = mlirLocationUnknownGet(context);
        let mut op_state = mlirOperationStateGet(
            mlirStringRefCreateFromCString(op_mnemonic.as_ptr()),
            location,
        );
        mlirOperationStateAddOperands(
            &mut op_state,
            operands.len().try_into().unwrap(),
            operands.as_ptr(),
        );
        mlirOperationStateAddAttributes(
            &mut op_state,
            attrs.len().try_into().unwrap(),
            attrs.as_ptr(),
        );
        if auto_result_type_inference {
            mlirOperationStateEnableResultTypeInference(&mut op_state);
        } else {
            mlirOperationStateAddResults(
                &mut op_state,
                results.len().try_into().unwrap(),
                results.as_ptr(),
            );
        }
        mlirOperationCreate(&mut op_state)
    }
}

pub fn create_addi_op(context: MlirContext, lhs: MlirValue, rhs: MlirValue) -> MlirOperation {
    create_op(context, "arith.addi", &[lhs, rhs], &[], &[], true)
}

pub fn create_ret_op(context: MlirContext, ret_value: MlirValue) -> MlirOperation {
    create_op(context, "func.return", &[ret_value], &[], &[], false)
}

pub fn create_constant_int_op(context: MlirContext, cst_value: i64, width: u32) -> MlirOperation {
    unsafe {
        let result_type = mlirIntegerTypeGet(context, width);
        let value_str = CString::new("value").unwrap();
        let value_attr = mlirNamedAttributeGet(
            mlirIdentifierGet(context, mlirStringRefCreateFromCString(value_str.as_ptr())),
            mlirIntegerAttrGet(result_type, cst_value),
        );
        create_op(
            context,
            "arith.constant",
            &[],
            &[result_type],
            &[value_attr],
            true,
        )
    }
}

pub fn create_constant_flat_tensor_op(
    context: MlirContext,
    cst_table: &[i64],
    bitwidth: u32,
) -> MlirOperation {
    let shape = [cst_table.len().try_into().unwrap()];
    create_constant_tensor_op(context, &shape, cst_table, bitwidth)
}

pub fn create_constant_tensor_op(
    context: MlirContext,
    shape: &[i64],
    cst_table: &[i64],
    bitwidth: u32,
) -> MlirOperation {
    unsafe {
        let result_type = mlirRankedTensorTypeGet(
            shape.len().try_into().unwrap(),
            shape.as_ptr(),
            mlirIntegerTypeGet(context, bitwidth),
            mlirAttributeGetNull(),
        );
        let cst_table_attrs: Vec<MlirAttribute> = cst_table
            .into_iter()
            .map(|value| mlirIntegerAttrGet(mlirIntegerTypeGet(context, bitwidth), *value))
            .collect();
        let value_str = CString::new("value").unwrap();
        let value_attr = mlirNamedAttributeGet(
            mlirIdentifierGet(context, mlirStringRefCreateFromCString(value_str.as_ptr())),
            mlirDenseElementsAttrGet(
                result_type,
                cst_table.len().try_into().unwrap(),
                cst_table_attrs.as_ptr(),
            ),
        );
        create_op(
            context,
            "arith.constant",
            &[],
            &[result_type],
            &[value_attr],
            true,
        )
    }
}

pub unsafe fn register_all_dialects(context: MlirContext) {
    let registry = mlirDialectRegistryCreate();
    mlirRegisterAllDialects(registry);
    mlirContextAppendDialectRegistry(context, registry);
    mlirContextLoadAllAvailableDialects(context);
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_function_type() {
        unsafe {
            let context = mlirContextCreate();
            let func_type = mlirFunctionTypeGet(context, 0, std::ptr::null(), 0, std::ptr::null());
            assert!(mlirTypeIsAFunction(func_type));
            mlirContextDestroy(context);
        }
    }

    #[test]
    fn test_module_parsing() {
        unsafe {
            let context = mlirContextCreate();
            register_all_dialects(context);
            let module_string = "
            module{
                func.func @test(%arg0: i64, %arg1: i64) -> i64 {
                    %1 = arith.addi %arg0, %arg1 : i64
                    return %1: i64
                }
            }";
            let module_cstring = CString::new(module_string).unwrap();
            let module_reference = mlirStringRefCreateFromCString(module_cstring.as_ptr());

            let parsed_module = mlirModuleCreateParse(context, module_reference);
            let parsed_func = mlirBlockGetFirstOperation(mlirModuleGetBody(parsed_module));

            let func_type_str = CString::new("function_type").unwrap();
            // just check that we do have a function here, which should be enough to know that parsing worked well
            assert!(mlirTypeIsAFunction(mlirTypeAttrGetValue(
                mlirOperationGetAttributeByName(
                    parsed_func,
                    mlirStringRefCreateFromCString(func_type_str.as_ptr()),
                )
            )));
        }
    }

    #[test]
    fn test_module_creation() {
        unsafe {
            let context = mlirContextCreate();
            register_all_dialects(context);

            // input/output types
            let func_input_types = [
                mlirIntegerTypeGet(context, 64),
                mlirIntegerTypeGet(context, 64),
            ];
            let func_output_types = [mlirIntegerTypeGet(context, 64)];

            let func_op = create_func_with_block(
                context,
                "test",
                func_input_types.as_slice(),
                func_output_types.as_slice(),
            );

            let func_block = mlirRegionGetFirstBlock(mlirOperationGetFirstRegion(func_op));
            let func_args = [
                mlirBlockGetArgument(func_block, 0),
                mlirBlockGetArgument(func_block, 1),
            ];
            // create addi operation and append it to the block
            let addi_op = create_addi_op(context, func_args[0], func_args[1]);
            mlirBlockAppendOwnedOperation(func_block, addi_op);

            // create ret operation and append it to the block
            let ret_op = create_ret_op(context, mlirOperationGetResult(addi_op, 0));
            mlirBlockAppendOwnedOperation(func_block, ret_op);

            // create module to hold the previously created function
            let location = mlirLocationUnknownGet(context);
            let module = mlirModuleCreateEmpty(location);
            mlirBlockAppendOwnedOperation(mlirModuleGetBody(module), func_op);

            let printed_module =
                super::print_mlir_operation_to_string(mlirModuleGetOperation(module));
            let expected_module = "\
module {
  func.func @test(%arg0: i64, %arg1: i64) -> i64 {
    %0 = arith.addi %arg0, %arg1 : i64
    return %0 : i64
  }
}
";
            assert_eq!(
                printed_module, expected_module,
                "left: \n{}, right: \n{}",
                printed_module, expected_module
            );
        }
    }

    #[test]
    fn test_constant_flat_tensor() {
        unsafe {
            let context = mlirContextCreate();
            register_all_dialects(context);

            // create a constant flat tensor
            let contant_flat_tensor_op = create_constant_flat_tensor_op(context, &[0, 1, 2, 3], 64);

            let printed_op = print_mlir_operation_to_string(contant_flat_tensor_op);
            let expected_op = "%cst = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi64>\n";
            assert_eq!(printed_op, expected_op);
        }
    }

    #[test]
    fn test_constant_tensor() {
        unsafe {
            let context = mlirContextCreate();
            register_all_dialects(context);

            // create a constant tensor
            let contant_tensor_op = create_constant_tensor_op(context, &[2, 2], &[0, 1, 2, 3], 64);

            let printed_op = print_mlir_operation_to_string(contant_tensor_op);
            let expected_op = "%cst = arith.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>\n";
            assert_eq!(printed_op, expected_op);
        }
    }

    #[test]
    fn test_constant_tensor_with_signle_elem() {
        unsafe {
            let context = mlirContextCreate();
            register_all_dialects(context);

            // create a constant tensor
            let contant_tensor_op = create_constant_tensor_op(context, &[2, 2], &[0], 7);

            let printed_op = print_mlir_operation_to_string(contant_tensor_op);
            let expected_op = "%cst = arith.constant dense<0> : tensor<2x2xi7>\n";
            assert_eq!(printed_op, expected_op);
        }
    }

    #[test]
    fn test_constant_int() {
        unsafe {
            let context = mlirContextCreate();
            register_all_dialects(context);

            // create a constant flat tensor
            let contant_int_op = create_constant_int_op(context, 73, 10);

            let printed_op = print_mlir_operation_to_string(contant_int_op);
            let expected_op = "%c73_i10 = arith.constant 73 : i10\n";
            assert_eq!(printed_op, expected_op);
        }
    }
}
