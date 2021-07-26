use concrete_commons::dispersion::Variance;

use crate::array::FfiArray;
use crate::FfiArray;

#[repr(C)]
pub struct VarianceArray(FfiArray<Variance>);

#[no_mangle]
pub extern "C" fn allocate_variance_array(value: Variance, len: usize) -> VarianceArray {
    VarianceArray(FfiArray::allocate(value, len))
}

#[no_mangle]
pub extern "C" fn free_variance_array(array: VarianceArray) {
    FfiArray::free(array.0);
}

#[no_mangle]
pub extern "C" fn get_variance(array: VarianceArray, elt: usize) -> Variance {
    array.0.as_slice()[elt]
}

#[no_mangle]
pub extern "C" fn set_variance(array: VarianceArray, elt: usize, value: Variance) {
    let mut array = array;
    *array.0.as_mut_slice()[elt] = value;
}

#[no_mangle]
pub extern "C" fn get_size(array: VarianceArray) -> usize {
    array.0.as_slice().len()
}
