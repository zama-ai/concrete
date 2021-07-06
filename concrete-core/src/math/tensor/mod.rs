//! Operations on collections of values.
//!
//! This module contains a [`Tensor`] type, central to the whole library. In essence, a tensor
//! wraps a collection container, and provides a set of methods to operate with other tensors of
//! the same length:
//! ```
//! use concrete_core::math::tensor::Tensor;
//! // We allocate two tensors of size 10
//! let mut tensor1 = Tensor::allocate(5. as f32, 10);
//! let tensor2 = Tensor::allocate(3. as f32, 10);
//! // We update the values of `tensor1` inplace, by adding it the values of `tensor2`;
//! tensor1.update_with_add(&tensor2);
//! ```
//!
//! The first interest of this type is that it can be backed by several collection containers,
//! such as `Vec<T>`, `&mut [T]` or `&[T]`. Operations can homogeneously be applied to tensors
//! backed by different containers:
//! ```
//! use concrete_core::math::tensor::Tensor;
//! // `allocate` returns a tensor backed by a `Vec`
//! let tensor1: Tensor<Vec<u32>> = Tensor::allocate(5, 100);
//! // `from_cont` allows you to create a tensor from any container
//! let mut distant_container = vec![4 as u32; 100];
//! let mut tensor2: Tensor<&mut [u32]> = Tensor::from_container(distant_container.as_mut_slice());
//! // We update the values of `distant_container` via `tensor2`
//! tensor2.update_with_add(&tensor1);
//! ```
//!
//! It is important to note that the `Tensor` type we have here, is *not* an n-dimmensional array,
//! as is common in scientific computating libraries. It is indexed by a single integral value,
//! and only operations with tensors of the same *length* are authorized.
//!
//! Despite this apparent limitation, `Tensor` are used throughout this library as the backbone of
//! several structures representing multi-dimensional collections. The pattern we use for such
//! structures is pretty simple:
//! ```
//! use concrete_core::math::tensor::{Tensor, AsRefTensor, AsRefSlice};
//!
//! // We want to have a matrix structure stored row-major.
//! pub struct Matrix<Cont>{
//!     tensor: Tensor<Cont>,
//!     row_length: usize,
//! }
//!
//! // Our matrix is row-major, so we must be able to iterate over rows.
//! pub struct Row<Cont>{
//!     tensor: Tensor<Cont>
//! }
//!
//! impl<Cont> Matrix<Cont> {
//!     
//!     // Returns an iterator over the matrix rows.
//!     pub fn row_iter(&self) -> impl Iterator<Item = Row<&[<Self as AsRefTensor>::Element]>>
//!     where
//!         Self: AsRefTensor
//!     {
//!         self.as_tensor()
//!             .as_slice()
//!             .chunks(self.row_length)
//!             .map(|sub| Row{tensor: Tensor::from_container(sub)})
//!     }
//! }
//! ```
//!
//! You can combine such structures to implement n-dimensional arrays of any size. This approach
//! has the benefit of making the orderinng of the element explicit, and to provide a specific type
//! and a specific set of operations for the different dimensions of your array. This prevents you
//! from shooting yourself in the foot when messing with your data layout, writing new code, or
//! refactoring.

// This macro implements various traits for a tensor-based object. To work properly, the object in
// question must be a structure with a `tensor` field.
#[doc(hidden)]
#[macro_export]
macro_rules! tensor_traits {
    ($Type:ident) => {
        impl<Element, Cont> $crate::math::tensor::AsRefTensor for $Type<Cont>
        where
            Cont: $crate::math::tensor::AsRefSlice<Element = Element>,
        {
            type Element = Element;
            type Container = Cont;
            fn as_tensor(&self) -> &Tensor<Self::Container> {
                &self.tensor
            }
        }

        impl<Element, Cont> $crate::math::tensor::AsMutTensor for $Type<Cont>
        where
            Cont: $crate::math::tensor::AsMutSlice<Element = Element>,
        {
            type Element = Element;
            type Container = Cont;
            fn as_mut_tensor(
                &mut self,
            ) -> &mut Tensor<<Self as $crate::math::tensor::AsMutTensor>::Container> {
                &mut self.tensor
            }
        }

        impl<Cont> $crate::math::tensor::IntoTensor for $Type<Cont>
        where
            Cont: $crate::math::tensor::AsRefSlice,
        {
            type Element = <Cont as $crate::math::tensor::AsRefSlice>::Element;
            type Container = Cont;
            fn into_tensor(self) -> Tensor<Self::Container> {
                self.tensor
            }
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! current_func_path {
    () => {{
        fn name<T>(_any: T) -> &'static str {
            std::any::type_name::<T>()
        }
        fn t() {}
        let output = name(t);
        &output[..output.len() - 3]
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! ck_dim_eq {
    ($tensor_size: expr => $($size: expr),* ) => {
        let func = $crate::current_func_path!();
        $(

            debug_assert!(
                $tensor_size == $size,
                "Called operation {} on tensors of incompatible size. {} (={:?}) does not equals \
                {} (={:?}).",
                func,
                stringify!($size),
                $size,
                stringify!($tensor_size),
                $tensor_size
            );
        )*
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! ck_dim_div {
    ($tensor_size: expr => $($size: expr),* ) => {
        $(
            let func = $crate::current_func_path!();
            debug_assert!(
                $tensor_size % $size == 0,
                "Called operation {} on tensors of incompatible size. {} (={:?}) does not divide \
                {} (={:?})",
                func,
                stringify!($size),
                $size,
                stringify!($tensor_size),
                $tensor_size
            );
        )*
    };
}

#[cfg(test)]
mod tests;

mod errors;
pub use errors::*;

#[allow(clippy::module_inception)]
mod tensor;
pub use tensor::*;

mod as_slice;
pub use as_slice::*;

mod as_element;
pub use as_element::*;

mod as_tensor;
pub use as_tensor::*;

mod into_tensor;
pub use into_tensor::*;
