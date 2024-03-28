# Modules

{% hint style="warning" %}
Modules are still experimental. They are only compatible with composition, that is, all function outputs can be used as inputs for every functions. The crypto-parameters used in this mode are pretty large, and will likely give slow execution time.
{% endhint %}

In some cases, it might be interesting to deploy a server that is able to execute different functions. With *Concrete*, it is possible to compile fhe _modules_, that can contain many different functions, at once. All the functions are compiled in a single step and can be [deployed with the same artifacts](../guides/deploy.md#deployment-of-modules). Here is an example:

```python
from concrete import fhe

@fhe.module()
class MyModule:
    @fhe.function({"x": "encrypted"})
    def inc(x):
        return x + 1 % 20

    @fhe.function({"x": "encrypted"})
    def dec(x):
        return x - 1 % 20
```

The `MyModule` fhe module can then be compiled using the `compile` method, by providing a dictionnary of input sets for every functions:

```python
inputset = list(range(20))
configuration = fhe.Configuration(
    parameter_selection_strategy="v0",
    composable=True,
)
my_module = MyModule.compile(
    {"inc": inputset, "dec": inputset},
    configuration,
```

{% hint style="warning" %}
Note that here we can see a current limitation of modules: The configuration must use the `parameter_selection_strategy` of `v0`, and activate the `composable` flag.
{% endhint %}

After the module has been compiled, we can encrypt and call the different functions in the following way:

```python
x = 5
x_enc = my_module.inc.encrypt(x)
x_inc_enc = my_module.inc.run(x_enc)
x_inc = my_module.inc.decrypt(x_inc_enc)
assert x_inc == 6

x_inc_dec_enc = my_module.dec.run(x_inc_enc)
x_inc_dec = my_module.dec.decrypt(x_inc_dec_enc)
assert x_inc_dec == 5

for _ in range(10):
    x_enc = my_module.inc.run(x_enc)
x_dec = my_module.inc.decrypt(x_enc)
assert x_dec == 15
```
