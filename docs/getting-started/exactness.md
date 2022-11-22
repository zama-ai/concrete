# Exactness

One of the most common operations in **Concrete-Numpy** is `Table Lookups` (TLUs). TLUs are performed with an FHE operation called `Programmable Bootstrapping` (PBS). PBSes have a certain probability of error, which, when triggered, result in inaccurate results.

Let's say you have the table:

```python
[0, 1, 4, 9, 16, 25, 36, 49, 64]
```

And you performed a table lookup using `4`. The result you should get is `16`, but because of the possibility of error, you can sometimes get `9` or `25`. Sometimes even `4` or `36` if you have a high probability of error.

The probability of this error can be configured through the `p_error` and `global_p_error` configuration options. The difference between these two options is that, `p_error` is for individual TLUs but `global_p_error` is for the whole circuit.

Here is an example, if you set `p_error` to `0.01`, it means every TLU in the circuit will have a 1% chance of not being exact and 99% chance of being exact. If you have a single TLU in the circuit, `global_p_error` would be 1% as well. But if you have 2 TLUs for example, `global_p_error` would be almost 2% (`1 - (0.99 * 0.99)`).

However, if you set `global_p_error` to `0.01`, the whole circuit will have 1% probability of being not exact, no matter how many table lookups are there.

If you set both of them, both will be satisfied. Essentially, the stricter one will be used.

By default, `p_error` is set to `None` and `global_p_error` is set to `1 / 100_000`. Feel free to play with these configuration options to pick the one best suited for your needs! For example, in some machine learning use cases, off-by-one or off-by-two errors doesn't affect the result much, in such cases `p_error` could be set to increase performance without losing accuracy.

See [How to Configure](../howto/configure.md) to learn how you can set a custom `p_error` and/or `global_p_error`.

{% hint style="info" %}
Configuring either of those variables would affect computation time (compilation, keys generation, circuit execution) and space requirements (size of the keys on disk and in memory). Lower error probability would result in longer computation time and larger space requirements. 
{% endhint %}
