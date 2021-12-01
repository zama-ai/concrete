//! A module containing some utilities.

/// This macro creates a string that represent the headline of the impl block being benchmarked.
/// It type checks the input such that there is no risk of misspelling. Though the error triggered
/// may be difficult to interpret ¯\_(ツ)_/¯.
macro_rules! benchmark_name {
    (impl $Trait:ident<$($Param:ty),+> for $Engine:ty) => {
        {
            let _type_check_signature: Box<dyn $Trait<$($Param),*, EngineError=<$Engine>::EngineError>> = Box::new(<$Engine>::new().unwrap());
            let mut output = String::from("impl ");
            output.push_str(format!("{}<", stringify!($Trait)).as_str());
            $(
            output.push_str(format!("{},", $crate::utils::type_name::<$Param>()).as_str());
            )*
            output.pop();
            output.push_str(format!("> for {}", $crate::utils::type_name::<$Engine>()).as_str());
            output
        }
    };
}
pub(crate) use benchmark_name;

/// A function returning the name of the type (just the name, not the path).
pub fn type_name<T>() -> &'static str {
    std::any::type_name::<T>()
        .split("::")
        .collect::<Vec<_>>()
        .pop()
        .unwrap()
}

/// A trait to create values and vec of values of integer numeric types.
pub trait IntegerValue: Sized {
    /// Returns a value of the integer type.
    fn any() -> Self;
    /// Returns a vec of values of the integer type.
    fn any_vec(size: usize) -> Vec<Self>;
}

impl IntegerValue for u32 {
    fn any() -> Self {
        1u32
    }

    fn any_vec(size: usize) -> Vec<Self> {
        vec![1u32; size]
    }
}

impl IntegerValue for u64 {
    fn any() -> Self {
        1u64
    }

    fn any_vec(size: usize) -> Vec<Self> {
        vec![1u64; size]
    }
}
