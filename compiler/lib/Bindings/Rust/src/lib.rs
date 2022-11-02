#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::ops::AddAssign;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

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
        mlirOperationPrint(
            op,
            Some(mlir_rust_string_receiver_callback),
            receiver_ptr
        );
    }

    rust_string
}


#[cfg(test)]
mod concrete_compiler_tests {
    use super::*;
    use std::ffi::CString;

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
            mlirRegisterAllDialects(context);
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
            mlirRegisterAllDialects(context);
            let location = mlirLocationUnknownGet(context);

            // input/output types
            let func_input_types = [
                mlirIntegerTypeGet(context, 64),
                mlirIntegerTypeGet(context, 64),
            ];
            let func_output_type = [mlirIntegerTypeGet(context, 64)];

            // create the main block of the function
            let func_block = mlirBlockCreate(2, func_input_types.as_ptr(), &location);

            let location = mlirLocationUnknownGet(context);
            // create addi operation and append it to the block
            let addi_str = CString::new("arith.addi").unwrap();
            let mut addi_op_state =
                mlirOperationStateGet(mlirStringRefCreateFromCString(addi_str.as_ptr()), location);
            mlirOperationStateAddOperands(
                &mut addi_op_state,
                2,
                [
                    mlirBlockGetArgument(func_block, 0),
                    mlirBlockGetArgument(func_block, 1),
                ]
                .as_ptr(),
            );
            mlirOperationStateEnableResultTypeInference(&mut addi_op_state);
            let addi_op = mlirOperationCreate(&mut addi_op_state);
            mlirBlockAppendOwnedOperation(func_block, addi_op);

            // create return operation and append it to the block
            let ret_str = CString::new("func.return").unwrap();
            let mut ret_op_state =
                mlirOperationStateGet(mlirStringRefCreateFromCString(ret_str.as_ptr()), location);
            mlirOperationStateAddOperands(
                &mut ret_op_state,
                1,
                [mlirOperationGetResult(addi_op, 0)].as_ptr(),
            );
            let ret_op = mlirOperationCreate(&mut ret_op_state);
            mlirBlockAppendOwnedOperation(func_block, ret_op);

            // create region to hold the previously created block
            let func_region = mlirRegionCreate();
            mlirRegionAppendOwnedBlock(func_region, func_block);

            // create function to hold the previously created region
            let func_str = CString::new("func.func").unwrap();
            let mut func_op_state =
                mlirOperationStateGet(mlirStringRefCreateFromCString(func_str.as_ptr()), location);
            mlirOperationStateAddOwnedRegions(&mut func_op_state, 1, [func_region].as_ptr());
            // set function attributes
            let func_type_str = CString::new("function_type").unwrap();
            let sym_name_str = CString::new("sym_name").unwrap();
            let func_name_str = CString::new("test").unwrap();
            let func_type_attr = mlirTypeAttrGet(mlirFunctionTypeGet(
                context,
                2,
                func_input_types.as_ptr(),
                1,
                func_output_type.as_ptr(),
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

            // create module to hold the previously created function
            let module = mlirModuleCreateEmpty(location);
            mlirBlockAppendOwnedOperation(mlirModuleGetBody(module), func_op);

            let printed_module = super::print_mlir_operation_to_string(mlirModuleGetOperation(module));
            let expected_module = "\
module {
  func.func @test(%arg0: i64, %arg1: i64) -> i64 {
    %0 = arith.addi %arg0, %arg1 : i64
    return %0 : i64
  }
}
";
            assert_eq!(
                printed_module,
                expected_module,
                "left: \n{}, right: \n{}",
                printed_module,
                expected_module);
        }
    }
}
