#[cfg(feature = "booleans")]
use crate::booleans::{BoolConfig, DynFheBoolEncryptor, FheBoolParameters};

#[cfg(feature = "shortints")]
use crate::shortints::{
    DynShortIntEncryptor, DynShortIntParameters, FheUint2Parameters, FheUint3Parameters,
    FheUint4Parameters, ShortIntConfig,
};

#[cfg(feature = "integers")]
use crate::integers::{DynIntegerEncryptor, DynIntegerParameters, IntegerConfig};

/// The config type
#[derive(Clone, Debug)]
pub struct Config {
    #[cfg(feature = "booleans")]
    pub(crate) bool_config: BoolConfig,
    #[cfg(feature = "shortints")]
    pub(crate) shortint_config: ShortIntConfig,
    #[cfg(feature = "integers")]
    pub(crate) integer_config: IntegerConfig,
}

/// The builder to create your config
///
/// This struct is what you will to use to build your
/// configuration.
///
/// # Why ?
///
/// The configuration is needed to select which types you are going to use or not
/// and which parameters you wish to use for these types (whether it is the default parameters or
/// some custom parameters).
///
/// To be able to configure a type, its "cargo feature kind" must be enabled (see the [table]).
///
/// The configuration is needed for the crate to be able to initialize and generate
/// all the needed client and server keys as well as other internal details.
///
/// As generating these keys and details for types that you are not going to use would be
/// a waste of time and space (both memory and disk if you serialize), generating a config is an
/// important step.
///
/// [table]: index.html#data-types
pub struct ConfigBuilder {
    config: Config,
}

impl ConfigBuilder {
    /// Create a new builder with all the data types activated with their default parameters
    pub fn all_enabled() -> Self {
        Self {
            config: Config {
                #[cfg(feature = "booleans")]
                bool_config: BoolConfig::all_default(),
                #[cfg(feature = "shortints")]
                shortint_config: ShortIntConfig::all_default(),
                #[cfg(feature = "integers")]
                integer_config: IntegerConfig::all_default(),
            },
        }
    }

    /// Create a new builder with all the data types disabled
    pub fn all_disabled() -> Self {
        Self {
            config: Config {
                #[cfg(feature = "booleans")]
                bool_config: BoolConfig::all_none(),
                #[cfg(feature = "shortints")]
                shortint_config: ShortIntConfig::all_none(),
                #[cfg(feature = "integers")]
                integer_config: IntegerConfig::all_none(),
            },
        }
    }

    /// Enables the [FheBool] type with default parameters
    ///
    /// [FheBool]: crate::FheBool
    #[cfg(feature = "booleans")]
    pub fn enable_default_bool(mut self) -> Self {
        self.config.bool_config.parameters = Some(FheBoolParameters::default());
        self
    }

    /// Enables the [FheBool] type with the given parameters
    ///
    /// [FheBool]: crate::FheBool
    #[cfg(feature = "booleans")]
    pub fn enable_custom_bool(mut self, params: FheBoolParameters) -> Self {
        self.config.bool_config.parameters = Some(params);
        self
    }

    /// Disables the [FheBool] type
    ///
    /// [FheBool]: crate::FheBool
    #[cfg(feature = "booleans")]
    pub fn disable_bool(mut self) -> Self {
        self.config.bool_config.parameters = None;
        self
    }

    /// Creates a new boolean type with the given parameters
    ///
    /// # Returns
    ///
    /// This returns the Encryptor of the new type
    #[cfg(feature = "booleans")]
    pub fn add_bool_type(&mut self, parameters: FheBoolParameters) -> DynFheBoolEncryptor {
        self.config.bool_config.add_bool_type(parameters)
    }

    /// Enables the [FheUint2] type with default parameters
    ///
    /// [FheUint2]: crate::FheUint2
    #[cfg(feature = "shortints")]
    pub fn enable_default_uint2(mut self) -> Self {
        self.config.shortint_config.uint2_parameters = Some(FheUint2Parameters::default());
        self
    }

    /// Enables the [FheUint2] type with the given parameters
    ///
    /// [FheUint2]: crate::FheUint2
    #[cfg(feature = "shortints")]
    pub fn enable_custom_uint2(mut self, params: FheUint2Parameters) -> Self {
        self.config.shortint_config.uint2_parameters = Some(params);
        self
    }

    /// Disables the [FheUint2] type
    ///
    /// [FheUint2]: crate::FheUint2
    #[cfg(feature = "shortints")]
    pub fn disable_uint2(mut self) -> Self {
        self.config.shortint_config.uint2_parameters = None;
        self
    }

    /// Enables the [FheUint3] type with default parameters
    ///
    /// [FheUint3]: crate::FheUint3
    #[cfg(feature = "shortints")]
    pub fn enable_default_uint3(mut self) -> Self {
        self.config.shortint_config.uint3_parameters = Some(FheUint3Parameters::default());
        self
    }

    /// Enables the [FheUint3] type with the given parameters
    ///
    /// [FheUint3]: crate::FheUint3
    #[cfg(feature = "shortints")]
    pub fn enable_custom_uint3(mut self, params: FheUint3Parameters) -> Self {
        self.config.shortint_config.uint3_parameters = Some(params);
        self
    }

    /// Disables the [FheUint3] type
    ///
    /// [FheUint3]: crate::FheUint3
    #[cfg(feature = "shortints")]
    pub fn disable_uint3(mut self) -> Self {
        self.config.shortint_config.uint3_parameters = None;
        self
    }

    /// Enables the [FheUint4] type with default parameters
    ///
    /// [FheUint4]: crate::FheUint4
    #[cfg(feature = "shortints")]
    pub fn enable_default_uint4(mut self) -> Self {
        self.config.shortint_config.uint4_parameters = Some(FheUint4Parameters::default());
        self
    }

    /// Enables the [FheUint4] type with the given parameters
    ///
    /// [FheUint4]: crate::FheUint4
    #[cfg(feature = "shortints")]
    pub fn enable_custom_uint4(mut self, params: FheUint4Parameters) -> Self {
        self.config.shortint_config.uint4_parameters = Some(params);
        self
    }

    /// Disables the [FheUint4] type
    ///
    /// [FheUint4]: crate::FheUint4
    #[cfg(feature = "shortints")]
    pub fn disable_uint4(mut self) -> Self {
        self.config.shortint_config.uint4_parameters = None;
        self
    }

    /// Creates a new short integer type with the given parameters
    ///
    /// # Returns
    ///
    /// This returns the Encryptor of the new type
    #[cfg(feature = "shortints")]
    pub fn add_short_int_type(
        &mut self,
        parameters: DynShortIntParameters,
    ) -> DynShortIntEncryptor {
        self.config.shortint_config.add_short_int_type(parameters)
    }

    #[cfg(feature = "integers")]
    pub fn enable_default_uint8(mut self) -> Self {
        self.config.integer_config.uint8_params = Some(Default::default());
        self
    }

    #[cfg(feature = "integers")]
    pub fn enable_custom_uint8(mut self, parameters: FheUint2Parameters) -> Self {
        self.config.integer_config.uint8_params = Some(parameters);
        self
    }

    #[cfg(feature = "integers")]
    pub fn disable_uint8(mut self) -> Self {
        self.config.integer_config.uint8_params = None;
        self
    }

    #[cfg(feature = "integers")]
    pub fn enable_default_uint12(mut self) -> Self {
        self.config.integer_config.uint12_params = Some(Default::default());
        self
    }

    #[cfg(feature = "integers")]
    pub fn enable_custom_uint12(mut self, parameters: FheUint2Parameters) -> Self {
        self.config.integer_config.uint12_params = Some(parameters);
        self
    }

    #[cfg(feature = "integers")]
    pub fn disable_uint12(mut self) -> Self {
        self.config.integer_config.uint12_params = None;
        self
    }

    #[cfg(feature = "integers")]
    pub fn enable_default_uint16(mut self) -> Self {
        self.config.integer_config.uint16_params = Some(Default::default());
        self
    }

    #[cfg(feature = "integers")]
    pub fn enable_custom_uint16(mut self, parameters: FheUint2Parameters) -> Self {
        self.config.integer_config.uint16_params = Some(parameters);
        self
    }

    #[cfg(feature = "integers")]
    pub fn disable_uint16(mut self) -> Self {
        self.config.integer_config.uint16_params = None;
        self
    }

    #[cfg(feature = "integers")]
    pub fn add_integer_type(&mut self, parameters: DynIntegerParameters) -> DynIntegerEncryptor {
        self.config.integer_config.add_integer_type(parameters)
    }

    pub fn build(self) -> Config {
        self.config
    }
}

impl From<ConfigBuilder> for Config {
    fn from(builder: ConfigBuilder) -> Self {
        builder.build()
    }
}
