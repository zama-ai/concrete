use crate::array::FfiArray;

#[repr(C)]
pub struct u32Array(FfiArray<u32>);

#[no_mangle]
pub extern "C" fn allocate_u32_array(value: u32, len: usize) -> u32Array {
    u32Array(FfiArray::allocate(value, len))
}

#[no_mangle]
pub extern "C" fn free_u32_array(array: u32Array) {
    FfiArray::free(array.0);
}

#[no_mangle]
pub extern "C" fn get_u32(array: u32Array, elt: usize) -> u32 {
    array.0.as_slice()[elt]
}

#[no_mangle]
pub extern "C" fn set_u32(array: u32Array, elt: usize, value: u32) {
    let mut array = array;
    *array.0.as_mut_slice()[elt] = value;
}

#[no_mangle]
pub extern "C" fn get_size(array: u32Array) -> usize {
    array.0.as_slice().len()
}
