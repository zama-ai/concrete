# Exactness

One of the most common operations in Concrete is `Table Lookups` (TLUs). TLUs are performed with an FHE operation called `Programmable Bootstrapping` (PBS). PBS's have a certain probability of error, which, when triggered, result in inaccurate results.

Let's say you have the table:

```python
lut = [0, 1, 4, 9, 16, 25, 36, 49, 64]
```

And you perform a Table Lookup using `4`. The result you should get is `lut[4] = 16`, but because of the possibility of error, you could get any other value in the table.

The probability of this error can be configured through the `p_error` and `global_p_error` configuration options. The difference between these two options is that, `p_error` is for individual TLUs but `global_p_error` is for the whole circuit.

If you set `p_error` to `0.01`, for example, it means every TLU in the circuit will have a 99% chance of being exact with a 1% probability of error. If you have a single TLU in the circuit, `global_p_error` would be 1% as well. But if you have 2 TLUs for example, `global_p_error` would be almost 2% (`1 - (0.99 * 0.99)`).

However, if you set `global_p_error` to `0.01`, the whole circuit will have 1% probability of error, no matter how many Table Lookups are included.

If you set both of them, both will be satisfied. Essentially, the stricter one will be used.

By default, both `p_error` and `global_p_error` is set to `None`, which results in a `global_p_error` of `1 / 100_000` being used.&#x20;

Feel free to play with these configuration options to pick the one best suited for your needs! See [How to Configure](../howto/configure.md) to learn how you can set a custom `p_error` and/or `global_p_error`.

{% hint style="info" %}
Configuring either of those variables impacts computation time (compilation, keys generation, circuit execution) and space requirements (size of the keys on disk and in memory). Lower error probabilities would result in longer computation times and larger space requirements.
{% endhint %}
