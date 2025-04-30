<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/__init__.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.tfhers`
tfhers module to represent, and compute on tfhers integer values. 

**Global Variables**
---------------
- **dtypes**
- **bridge**
- **specs**
- **values**
- **tracing**

---

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/__init__.py#L27"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_type_from_params`

```python
get_type_from_params(
    path_to_params_json: str,
    is_signed: bool,
    precision: int
) → TFHERSIntegerType
```

Get a TFHE-rs integer type from TFHE-rs parameters in JSON format. 



**Args:**
 
 - <b>`path_to_params_json`</b> (str):  path to the TFHE-rs parameters (JSON format) 
 - <b>`is_signed`</b> (bool):  sign of the result type 
 - <b>`precision`</b> (int):  precision of the result type 



**Returns:**
 
 - <b>`TFHERSIntegerType`</b>:  constructed type from the loaded parameters 


---

<a href="../../frontends/concrete-python/concrete/fhe/tfhers/__init__.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_type_from_params_dict`

```python
get_type_from_params_dict(
    crypto_param_dict: dict,
    is_signed: bool,
    precision: int
) → TFHERSIntegerType
```

Get a TFHE-rs integer type from TFHE-rs parameters in JSON format. 



**Args:**
 
 - <b>`crypto_param_dict`</b> (Dict):  dictionary of TFHE-rs parameters 
 - <b>`is_signed`</b> (bool):  sign of the result type 
 - <b>`precision`</b> (int):  precision of the result type 



**Returns:**
 
 - <b>`TFHERSIntegerType`</b>:  constructed type from the loaded parameters 


