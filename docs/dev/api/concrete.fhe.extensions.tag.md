<!-- markdownlint-disable -->

<a href="../../frontends/concrete-python/concrete/fhe/extensions/tag.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.fhe.extensions.tag`
Declaration of `tag` context manager, to allow tagging certain nodes. 


---

<a href="../../concrete/fhe/extensions/tag/tag#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `tag`

```python
tag(name: str)
```

Introduce a new tag to the tag stack. 

Can be nested, and the resulting tag will be `tag1.tag2`. 


