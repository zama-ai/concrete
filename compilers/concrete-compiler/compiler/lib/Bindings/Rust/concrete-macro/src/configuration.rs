use serde::{Deserialize, Deserializer};
use serde_json::Value;

struct PythonPickledEnum {
    py_type: String,
    py_tuple: Value,
}

impl<'de> Deserialize<'de> for PythonPickledEnum {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let json = Value::deserialize(deserializer)?;
        let Value::Object(obj) = json else {
            return Err(<D::Error as serde::de::Error>::custom("Missing object"));
        };
        let Some(py_reduce) = obj.get("py/reduce") else {
            return Err(<D::Error as serde::de::Error>::custom(
                "Missing field \"py/reduce\"",
            ));
        };
        let Value::Array(arr) = py_reduce else {
            return Err(<D::Error as serde::de::Error>::custom("Missing array"));
        };
        if arr.len() != 2 {
            return Err(<D::Error as serde::de::Error>::custom("Unexpected py_reduce array length"));
        }
        let Some(Value::Object(py_type_obj)) = arr.get(0) else {
            return Err(<D::Error as serde::de::Error>::custom("Missing object"));
        };
        let Some(Value::String(py_type)) = py_type_obj.get("py/type") else {
            return Err(<D::Error as serde::de::Error>::custom(
                "Missing \"py/type\" field.",
            ));
        };
        let Some(Value::Object(py_tuple_obj)) = arr.get(1) else {
            return Err(<D::Error as serde::de::Error>::custom("Missing object"));
        };
        let Some(py_tuple_value) = py_tuple_obj.get("py/tuple") else {
            return Err(<D::Error as serde::de::Error>::custom(
                "Missing \"py/tuple\" field.",
            ));
        };
        let Value::Array(py_tuple_arr) = py_tuple_value else {
            return Err(<D::Error as serde::de::Error>::custom(
                "\"py/tuple\" is not an array.",
            ));
        };
        if py_tuple_arr.len() != 1 {
            return Err(<D::Error as serde::de::Error>::custom("Unexpected py_tuple array length"));
        }
        let py_tuple = py_tuple_arr.get(0).unwrap();
        Ok(PythonPickledEnum {
            py_type: py_type.clone(),
            py_tuple: py_tuple.clone(),
        })
    }
}

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

#[derive(Deserialize, Debug)]
pub struct Configuration {
    show_optimizer: Option<bool>,
    loop_parallelize: bool,
    dataflow_parallelize: bool,
    auto_parallelize: bool,
    compress_evaluation_keys: bool,
    compress_input_ciphertexts: bool,
    p_error: Option<f64>,
    global_p_error: Option<f64>,
    parameter_selection_strategy: ParameterSelectionStrategy,
    multi_parameter_strategy: MultiParameterStrategy,
    enable_tlu_fusing: bool,
    print_tlu_fusing: bool,
    detect_overflow_in_simulation: bool,
    composable: bool,
    // range_restriction: Option<RangeRestriction>,
    // keyset_restriction: Option<KeysetRestriction>,
    security_level: SecurityLevel,
}
