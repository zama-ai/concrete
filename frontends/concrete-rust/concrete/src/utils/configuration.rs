use super::python::{PythonPickledEnum, PythonPickledObject};
use serde::{Deserialize, Deserializer};
use serde_json::Value;

#[derive(Debug)]
pub enum ParameterSelectionStrategy {
    V0,
    Mono,
    Multi,
}

impl<'de> Deserialize<'de> for ParameterSelectionStrategy {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let python_enum = PythonPickledEnum::deserialize(deserializer)?;
        if python_enum.py_type.split(".").last() != Some("ParameterSelectionStrategy") {
            return Err(<D::Error as serde::de::Error>::custom(
                "Unexpected py/type.",
            ));
        }
        let Value::String(variant) = python_enum.py_tuple else {
            return Err(<D::Error as serde::de::Error>::custom(
                "Unexpected py/tuple.",
            ));
        };
        match variant.as_str() {
            "v0" => Ok(ParameterSelectionStrategy::V0),
            "mono" => Ok(ParameterSelectionStrategy::Mono),
            "multi" => Ok(ParameterSelectionStrategy::Multi),
            _ => Err(<D::Error as serde::de::Error>::custom(
                "Unexpected variant.",
            )),
        }
    }
}

#[derive(Debug)]
pub enum MultiParameterStrategy {
    Precision,
    PrecisionAndNorm2,
}

impl<'de> Deserialize<'de> for MultiParameterStrategy {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let python_enum = PythonPickledEnum::deserialize(deserializer)?;
        if python_enum.py_type.split(".").last() != Some("MultiParameterStrategy") {
            return Err(<D::Error as serde::de::Error>::custom(
                "Unexpected py/type.",
            ));
        }
        let Value::String(variant) = python_enum.py_tuple else {
            return Err(<D::Error as serde::de::Error>::custom(
                "Unexpected py/tuple.",
            ));
        };
        match variant.as_str() {
            "precision" => Ok(MultiParameterStrategy::Precision),
            "precision_and_norm2" => Ok(MultiParameterStrategy::PrecisionAndNorm2),
            _ => Err(<D::Error as serde::de::Error>::custom(
                "Unexpected variant.",
            )),
        }
    }
}

#[derive(Debug)]
pub enum SecurityLevel {
    Security128Bits,
    Security132Bits,
}

impl<'de> Deserialize<'de> for SecurityLevel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let python_enum = PythonPickledEnum::deserialize(deserializer)?;
        if python_enum.py_type.split(".").last() != Some("SecurityLevel") {
            return Err(<D::Error as serde::de::Error>::custom(
                "Unexpected py/type.",
            ));
        }
        let Value::Number(variant) = python_enum.py_tuple else {
            return Err(<D::Error as serde::de::Error>::custom(
                "Unexpected py/tuple.",
            ));
        };
        match variant.as_u64() {
            Some(128) => Ok(SecurityLevel::Security128Bits),
            Some(132) => Ok(SecurityLevel::Security132Bits),
            _ => Err(<D::Error as serde::de::Error>::custom(
                "Unexpected variant.",
            )),
        }
    }
}

#[derive(Debug)]
pub struct RangeRestriction(pub String);

impl<'de> Deserialize<'de> for RangeRestriction {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let python_object = PythonPickledObject::deserialize(deserializer)?;
        if python_object.py_object.split(".").last() != Some("RangeRestriction") {
            return Err(<D::Error as serde::de::Error>::custom(
                "Unexpected py/object.",
            ));
        }
        Ok(RangeRestriction(python_object.py_serialized))
    }
}

#[derive(Debug)]
pub struct KeysetRestriction(pub String);

impl<'de> Deserialize<'de> for KeysetRestriction {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let python_object = PythonPickledObject::deserialize(deserializer)?;
        if python_object.py_object.split(".").last() != Some("KeysetRestriction") {
            return Err(<D::Error as serde::de::Error>::custom(
                "Unexpected py/object.",
            ));
        }
        Ok(KeysetRestriction(python_object.py_serialized))
    }
}

#[derive(Deserialize, Debug)]
pub struct Configuration {
    pub show_optimizer: Option<bool>,
    pub loop_parallelize: bool,
    pub dataflow_parallelize: bool,
    pub auto_parallelize: bool,
    pub compress_evaluation_keys: bool,
    pub compress_input_ciphertexts: bool,
    pub p_error: Option<f64>,
    pub global_p_error: Option<f64>,
    pub parameter_selection_strategy: ParameterSelectionStrategy,
    pub multi_parameter_strategy: MultiParameterStrategy,
    pub enable_tlu_fusing: bool,
    pub detect_overflow_in_simulation: bool,
    pub composable: bool,
    pub range_restriction: Option<RangeRestriction>,
    pub keyset_restriction: Option<KeysetRestriction>,
    pub security_level: SecurityLevel,
}
