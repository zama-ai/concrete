use cxx::UniquePtr;

use crate::ffi::{GetTensor, Tensor, Value};

pub trait FromValue {
    type Spec;
    fn from_value(s: Self::Spec, v: UniquePtr<Value>) -> Self;
}

impl<T> FromValue for Tensor<T>
where
    Value: GetTensor<T>,
{
    type Spec = ();

    fn from_value(_s: Self::Spec, v: UniquePtr<Value>) -> Self {
        v.get_tensor().unwrap()
    }
}
