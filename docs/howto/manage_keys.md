# Manage Keys

**Concrete** generates keys for you implicitly when they are needed and if they have not already been generated. This is useful for development, but it's not flexible **(or secure!)** for production. Explicit key management API is introduced to be used in such cases to easily generate and re-use keys.

## Definition

Let's start by defining a circuit:

```python
from concrete import fhe

@fhe.compiler({"x": "encrypted"})
def f(x):
    return x ** 2

inputset = range(10)
circuit = f.compile(inputset)
```

Circuits have a property called `keys` of type `fhe.Keys`, which has several utility functions dedicated to key management!

## Generation

To explicitly generate keys for a circuit, you can use:

```python
circuit.keys.generate()
```

{% hint style="info" %}
Generated keys are stored in memory upon generation, unencrypted.
{% endhint %}

And it's possible to set a custom seed for reproducibility:

```python
circuit.keys.generate(seed=420)
```

{% hint style="warning" %}
Do not specify the seed manually in a production environment!
{% endhint %}

## Serialization

To serialize keys, say to send it across the network:

```python
serialized_keys: bytes = circuit.keys.serialize()
```

{% hint style="warning" %}
Keys are not serialized in encrypted form! Please make sure you keep them in a safe environment, or encrypt them manually after serialization.
{% endhint %}

## Deserialization

To deserialize the keys back, after receiving serialized keys:

```python
keys: fhe.Keys = fhe.Keys.deserialize(serialized_keys)
```

## Assignment

Once you have a valid `fhe.Keys` object, you can directly assign it to the circuit:

```python
circuit.keys = keys
```

{% hint style="warning" %}
If assigned keys are generated for a different circuit, an exception will be raised.
{% endhint %}

## Saving

You can also use the filesystem to store the keys directly, without needing to deal with serialization and file management yourself:

```python
circuit.keys.save("/path/to/keys")
```

{% hint style="warning" %}
Keys are not saved encrypted! Please make sure you store them in a safe environment, or encrypt them manually after saving.
{% endhint %}

## Loading

After keys are saved to disk, you can load them back via:

```python
circuit.keys.load("/path/to/keys")
```

## Automatic Management

If you want to generate keys in the first run and reuse the keys in consecutive runs:

```python
circuit.keys.load_if_exists_generate_and_save_otherwise("/path/to/keys")
```
