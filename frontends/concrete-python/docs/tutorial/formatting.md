# Formatting

You can convert your compiled circuit into its textual representation by converting it to string:

<!--pytest-codeblocks:skip-->
```python
str(circuit)
```

If you just want to see the output on your terminal, you can directly print it as well:

<!--pytest-codeblocks:skip-->
```python
print(circuit)
```

{% hint style="warning" %}
Formatting is just for debugging. It's not possible to serialize the circuit back from its textual representation. See [How to Deploy](../howto/deploy.md) if that's your goal.
{% endhint %}
