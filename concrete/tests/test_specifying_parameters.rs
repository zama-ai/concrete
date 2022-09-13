use concrete::parameters::*;
use concrete::ConfigBuilder;
#[cfg(feature = "booleans")]
use concrete::FheBoolParameters;
#[cfg(feature = "shortints")]
use concrete::{FheUint2Parameters, FheUint3Parameters, FheUint4Parameters};

/// We want user to be able to specify the parameters for each type.
///
/// As each parameter struct of each intermediate lib uses a new type
/// we have to make sure these new types are usable by the user.
///
/// We have to make sure that the user can give its parameters no matter the
/// features enabled.
///
/// We do not care about correctness of values in this test,
/// only that is compiles.
#[test]
fn test_user_is_able_to_specify_parameters() {
    let config = ConfigBuilder::all_disabled();

    #[cfg(feature = "booleans")]
    let config = {
        config.enable_custom_bool(FheBoolParameters {
            lwe_dimension: LweDimension(1),
            glwe_dimension: GlweDimension(1),
            polynomial_size: PolynomialSize(1),
            lwe_modular_std_dev: StandardDev(1.0),
            glwe_modular_std_dev: StandardDev(1.0),
            pbs_base_log: DecompositionBaseLog(1),
            pbs_level: DecompositionLevelCount(1),
            ks_base_log: DecompositionBaseLog(1),
            ks_level: DecompositionLevelCount(1),
        })
    };

    #[cfg(feature = "booleans")]
    let config = {
        let mut config = config;
        let _instanciator = config.add_bool_type(FheBoolParameters {
            lwe_dimension: LweDimension(1),
            glwe_dimension: GlweDimension(1),
            polynomial_size: PolynomialSize(1),
            lwe_modular_std_dev: StandardDev(1.0),
            glwe_modular_std_dev: StandardDev(1.0),
            pbs_base_log: DecompositionBaseLog(1),
            pbs_level: DecompositionLevelCount(1),
            ks_base_log: DecompositionBaseLog(1),
            ks_level: DecompositionLevelCount(1),
        });
        config
    };

    #[cfg(feature = "shortints")]
    let config = {
        config
            .enable_custom_uint3(FheUint3Parameters {
                lwe_dimension: LweDimension(1),
                glwe_dimension: GlweDimension(1),
                polynomial_size: PolynomialSize(1),
                lwe_modular_std_dev: StandardDev(1.0),
                glwe_modular_std_dev: StandardDev(1.0),
                pbs_base_log: DecompositionBaseLog(1),
                pbs_level: DecompositionLevelCount(1),
                ks_base_log: DecompositionBaseLog(1),
                ks_level: DecompositionLevelCount(1),
                pfks_level: DecompositionLevelCount(1),
                pfks_base_log: DecompositionBaseLog(1),
                pfks_modular_std_dev: StandardDev(1.0),
                cbs_level: DecompositionLevelCount(1),
                cbs_base_log: DecompositionBaseLog(1),
                carry_modulus: CarryModulus(1),
            })
            .enable_custom_uint2(FheUint2Parameters {
                lwe_dimension: LweDimension(1),
                glwe_dimension: GlweDimension(1),
                polynomial_size: PolynomialSize(1),
                lwe_modular_std_dev: StandardDev(1.0),
                glwe_modular_std_dev: StandardDev(1.0),
                pbs_base_log: DecompositionBaseLog(1),
                pbs_level: DecompositionLevelCount(1),
                ks_base_log: DecompositionBaseLog(1),
                ks_level: DecompositionLevelCount(1),
                pfks_level: DecompositionLevelCount(1),
                pfks_base_log: DecompositionBaseLog(1),
                pfks_modular_std_dev: StandardDev(1.0),
                cbs_level: DecompositionLevelCount(1),
                cbs_base_log: DecompositionBaseLog(1),
                carry_modulus: CarryModulus(1),
            })
            .enable_custom_uint4(FheUint4Parameters {
                lwe_dimension: LweDimension(1),
                glwe_dimension: GlweDimension(1),
                polynomial_size: PolynomialSize(1),
                lwe_modular_std_dev: StandardDev(1.0),
                glwe_modular_std_dev: StandardDev(1.0),
                pbs_base_log: DecompositionBaseLog(1),
                pbs_level: DecompositionLevelCount(1),
                ks_base_log: DecompositionBaseLog(1),
                ks_level: DecompositionLevelCount(1),
                pfks_level: DecompositionLevelCount(1),
                pfks_base_log: DecompositionBaseLog(1),
                pfks_modular_std_dev: StandardDev(1.0),
                cbs_level: DecompositionLevelCount(1),
                cbs_base_log: DecompositionBaseLog(1),
                carry_modulus: CarryModulus(1),
            })
    };

    #[cfg(feature = "shortints")]
    let config = {
        use concrete::DynShortIntParameters;
        let mut config = config;
        let _instanciator = config.add_short_int_type(DynShortIntParameters {
            lwe_dimension: LweDimension(1),
            glwe_dimension: GlweDimension(1),
            polynomial_size: PolynomialSize(1),
            lwe_modular_std_dev: StandardDev(1.0),
            glwe_modular_std_dev: StandardDev(1.0),
            pbs_base_log: DecompositionBaseLog(1),
            pbs_level: DecompositionLevelCount(1),
            ks_base_log: DecompositionBaseLog(1),
            ks_level: DecompositionLevelCount(1),
            message_modulus: MessageModulus(2),
            pfks_level: DecompositionLevelCount(1),
            pfks_base_log: DecompositionBaseLog(1),
            pfks_modular_std_dev: StandardDev(1.0),
            cbs_level: DecompositionLevelCount(1),
            cbs_base_log: DecompositionBaseLog(1),
            carry_modulus: CarryModulus(2),
        });
        config
    };

    let _ = config.build();
}
