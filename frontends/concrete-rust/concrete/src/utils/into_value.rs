use crate::ffi::{Tensor, Value};
use cxx::UniquePtr;

pub trait IntoValue {
    fn into_value(self) -> UniquePtr<Value>;
}

impl<T> IntoValue for Tensor<T> {
    fn into_value(self) -> UniquePtr<Value> {
        Value::from_tensor(self)
    }
}
