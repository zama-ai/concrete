<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/dtypes/utils.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.dtypes.utils`
Declaration of various functions and constants related to data types. 

**Global Variables**
---------------
- **SignedInteger**
- **UnsignedInteger**

---

<a href="../../frontends/concrete-python/concrete/fhe/dtypes/utils.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `combine_dtypes`

```python
combine_dtypes(dtypes: List[BaseDataType]) → BaseDataType
```

Get the 'BaseDataType' that can represent a set of 'BaseDataType's. 



**Args:**
  dtypes (List[BaseDataType]):  dtypes to combine 



**Returns:**
  BaseDataType:  dtype that can hold all the given dtypes (potentially lossy) 


