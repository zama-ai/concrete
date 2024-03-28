# Modules

{% hint style="warning" %}
Modules are still experimental. They are only compatible with [composition](../compilation/composition.md), which means the outputs of every functions can be used directly as inputs for other functions. The crypto-parameters used in this mode are large and thus, the execution is likely to slow.
{% endhint %}

In some cases, deploying a server that can execute different functions is useful. *Concrete* can compile FHE _modules_, that can contain many different functions to execute at once. All the functions are compiled in a single step and can be [deployed with the same artifacts](../guides/deploy.md#deployment-of-modules). Here is an example:

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

You can compile the FHE module `MyModule` using the `compile` method. To do that, you need to provide a dictionnary of input sets for every function:

```python
inputset = list(range(20))
my_module = MyModule.compile({"inc": inputset, "dec": inputset})
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
