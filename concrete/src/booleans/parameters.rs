use concrete_boolean::parameters::BooleanParameters;

pub trait BooleanParameterSet: Into<BooleanParameters> {}

impl BooleanParameterSet for BooleanParameters {}

#[cfg(feature = "__newer_booleans")]
pub use new::FheBoolParameters;
#[cfg(not(feature = "__newer_booleans"))]
pub use old::FheBoolParameters;

#[cfg(not(feature = "__newer_booleans"))]
mod old {
    use concrete_boolean::parameters::BooleanParameters;
    pub use concrete_boolean::parameters::{DEFAULT_PARAMETERS, TFHE_LIB_PARAMETERS};
    #[cfg(feature = "serde")]
    use serde::{Deserialize, Serialize};

    #[cfg_attr(doc, cfg(feature = "booleans"))]
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    #[derive(Clone, Debug, PartialEq)]
    pub struct FheBoolParameters(pub(crate) BooleanParameters);

    impl FheBoolParameters {
        pub fn concrete_lib() -> Self {
            Self::from_static(&DEFAULT_PARAMETERS)
        }

        pub fn tfhe_lib() -> Self {
            Self::from_static(&TFHE_LIB_PARAMETERS)
        }

        fn from_static(params: &'static BooleanParameters) -> Self {
            Self(params.clone())
        }
    }

    impl From<FheBoolParameters> for BooleanParameters {
        fn from(params: FheBoolParameters) -> Self {
            params.0
        }
    }

    impl From<BooleanParameters> for FheBoolParameters {
        fn from(params: BooleanParameters) -> FheBoolParameters {
            Self(params)
        }
    }

    impl Default for FheBoolParameters {
        fn default() -> Self {
            Self::concrete_lib()
        }
    }
}

#[cfg(feature = "__newer_booleans")]
mod new {
    use concrete_boolean::parameters::{
        BooleanParameters, DecompositionBaseLog, DecompositionLevelCount, GlweDimension,
        LweDimension, PolynomialSize, StandardDev,
    };
    pub use concrete_boolean::parameters::{DEFAULT_PARAMETERS, TFHE_LIB_PARAMETERS};
    #[cfg(feature = "serde")]
    use serde::{Deserialize, Serialize};

    /// Parameters for [FheBool].
    #[cfg_attr(doc, cfg(feature = "booleans"))]
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    #[derive(Copy, Clone, Debug, PartialEq)]
    pub struct FheBoolParameters {
        pub lwe_dimension: LweDimension,
        pub glwe_dimension: GlweDimension,
        pub polynomial_size: PolynomialSize,
        pub lwe_modular_std_dev: StandardDev,
        pub glwe_modular_std_dev: StandardDev,
        pub pbs_base_log: DecompositionBaseLog,
        pub pbs_level: DecompositionLevelCount,
        pub ks_base_log: DecompositionBaseLog,
        pub ks_level: DecompositionLevelCount,
    }

    impl FheBoolParameters {
        pub fn concrete_lib() -> Self {
            Self::from_static(&DEFAULT_PARAMETERS)
        }

        pub fn tfhe_lib() -> Self {
            Self::from_static(&TFHE_LIB_PARAMETERS)
        }

        fn from_static(params: &'static BooleanParameters) -> Self {
            (*params).into()
        }
    }

    #[cfg(feature = "__newer_booleans")]
    impl From<FheBoolParameters> for BooleanParameters {
        fn from(params: FheBoolParameters) -> Self {
            Self {
                lwe_dimension: params.lwe_dimension,
                glwe_dimension: params.glwe_dimension,
                polynomial_size: params.polynomial_size,
                lwe_modular_std_dev: params.lwe_modular_std_dev,
                glwe_modular_std_dev: params.glwe_modular_std_dev,
                pbs_base_log: params.pbs_base_log,
                pbs_level: params.pbs_level,
                ks_base_log: params.ks_base_log,
                ks_level: params.ks_level,
            }
        }
    }

    impl From<BooleanParameters> for FheBoolParameters {
        fn from(params: BooleanParameters) -> FheBoolParameters {
            Self {
                lwe_dimension: params.lwe_dimension,
                glwe_dimension: params.glwe_dimension,
                polynomial_size: params.polynomial_size,
                lwe_modular_std_dev: params.lwe_modular_std_dev,
                glwe_modular_std_dev: params.glwe_modular_std_dev,
                pbs_base_log: params.pbs_base_log,
                pbs_level: params.pbs_level,
                ks_base_log: params.ks_base_log,
                ks_level: params.ks_level,
            }
        }
    }

    impl Default for FheBoolParameters {
        fn default() -> Self {
            Self::concrete_lib()
        }
    }
}
