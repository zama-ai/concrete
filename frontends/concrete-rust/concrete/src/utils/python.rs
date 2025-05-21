use serde::{Deserialize, Deserializer};
use serde_json::Value;

pub struct PythonPickledObject {
    pub py_object: String,
    pub py_serialized: String,
}

impl<'de> Deserialize<'de> for PythonPickledObject {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let json = Value::deserialize(deserializer)?;
        let Value::Object(obj) = json else {
            return Err(<D::Error as serde::de::Error>::custom("Missing object"));
        };
        let Some(Value::String(py_object)) = obj.get("py/object") else {
            return Err(<D::Error as serde::de::Error>::custom(
                "Missing field \"py/object\"",
            ));
        };
        let Some(py_serialized) = obj.get("serialized") else {
            return Err(<D::Error as serde::de::Error>::custom(
                "Missing field \"serialized\"",
            ));
        };
        Ok(PythonPickledObject {
            py_object: py_object.clone(),
            py_serialized: py_serialized.to_string(),
        })
    }
}

pub struct PythonPickledEnum {
    pub py_type: String,
    pub py_tuple: Value,
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
            return Err(<D::Error as serde::de::Error>::custom(
                "Unexpected py_reduce array length",
            ));
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
            return Err(<D::Error as serde::de::Error>::custom(
                "Unexpected py_tuple array length",
            ));
        }
        let py_tuple = py_tuple_arr.get(0).unwrap();
        Ok(PythonPickledEnum {
            py_type: py_type.clone(),
            py_tuple: py_tuple.clone(),
        })
    }
}
