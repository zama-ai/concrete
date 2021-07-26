use crate::array::FfiArray;

#[repr(C)]
pub struct u64Array(FfiArray<u64>);

#[no_mangle]
pub extern "C" fn allocate_u64_array(value: u64, len: usize) -> u64Array {
    u64Array(FfiArray::allocate(value, len))
}

#[no_mangle]
pub extern "C" fn free_u64_array(array: u64Array) {
    FfiArray::free(array.0);
}

#[no_mangle]
pub extern "C" fn get_u64(array: u64Array, elt: usize) -> u64 {
    array.0.as_slice()[elt]
}

#[no_mangle]
pub extern "C" fn set_u64(array: u64Array, elt: usize, value: u64) {
    let mut array = array;
    *array.0.as_mut_slice()[elt] = value;
}

#[no_mangle]
pub extern "C" fn get_size(array: u64Array) -> usize {
    array.0.as_slice().len()
}
