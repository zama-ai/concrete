use std::fmt::{Display, Formatter};

/// Unwrap 'Extension' trait
///
/// The goal of this trait is to add a method similar to `unwrap` to `Result<T, E>`
/// that uses the implementation of `Display` and not `Debug` as the
/// message in the panic.
pub trait UnwrapResultExt<T> {
    fn unwrap_display(self) -> T;
}

impl<T, E> UnwrapResultExt<T> for Result<T, E>
where
    E: Display,
{
    #[track_caller]
    fn unwrap_display(self) -> T {
        match self {
            Ok(t) => t,
            Err(e) => panic!("{}", e),
        }
    }
}

/// Enum that lists types available
///
/// Mainly used to provide good errors.
#[derive(Debug)]
pub enum Type {
    #[cfg(feature = "booleans")]
    FheBool,
    #[cfg(feature = "booleans")]
    DynamicBool,
    #[cfg(feature = "shortints")]
    FheUint2,
    #[cfg(feature = "shortints")]
    FheUint3,
    #[cfg(feature = "shortints")]
    FheUint4,
    #[cfg(feature = "shortints")]
    DynamicShortInt,
    #[cfg(feature = "integers")]
    FheUint8,
    #[cfg(feature = "integers")]
    FheUint12,
    #[cfg(feature = "integers")]
    FheUint16,
    #[cfg(feature = "integers")]
    DynamicInteger,
}

/// The server key of a given type was not initialized
#[derive(Debug)]
pub struct UninitializedServerKey(pub(crate) Type);

impl Display for UninitializedServerKey {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "The server key for the type '{:?}' was not properly initialized",
            self.0
        )
    }
}

impl std::error::Error for UninitializedServerKey {}

/// The client key of a given type was not initialized
#[derive(Debug)]
pub struct UninitializedClientKey(pub(crate) Type);

impl Display for UninitializedClientKey {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "The client key for the type '{:?}' was not properly initialized",
            self.0
        )
    }
}

impl std::error::Error for UninitializedClientKey {}

/// Error when trying to create a short integer from a value that was too big to be represented
#[derive(Debug, Eq, PartialEq)]
pub struct OutOfRangeError;

impl Display for OutOfRangeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Value is out of range")
    }
}

impl std::error::Error for OutOfRangeError {}
