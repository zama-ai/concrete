# Manage Keys
This document explains how to manage keys when using **Concrete**, introducing the key management API for generating, reusing, and securely handling keys.

**Concrete** generates keys lazily when needed. While this is convenient for development, it's not ideal for the production environment. The explicit key management API is available for you to easily generate and reuse keys as needed.

## Definition

Let's start by defining a circuit with the following example:

```python
from concrete import fhe

@fhe.compiler({"x": "encrypted"})
def f(x):
    return x ** 2

inputset = range(10)
circuit = f.compile(inputset)
```

Circuits have a `keys` property of type `fhe.Keys`, which includes several utilities for key management.

## Generation

To explicitly generate keys for a circuit, use:

```python
circuit.keys.generate()
```

{% hint style="info" %}
Generated keys are stored in memory and remain unencrypted.
{% endhint %}

You can also set a custom seed for reproducibility:

```python
circuit.keys.generate(seed=420)
```

{% hint style="warning" %}
Do not specify the seed manually in a production environment! This is not secure and should only be done for debugging purposes.
{% endhint %}

## Serialization

To serialize keys, for tasks such as sending them across a network, use:


```python
serialized_keys: bytes = circuit.keys.serialize()
```

{% hint style="warning" %}
Keys are not serialized in encrypted form. Please make sure you keep them in a safe environment, or encrypt them manually after serialization.
{% endhint %}

## Deserialization

To deserialize the keys back after receiving serialized keys, use:

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

You can also use the filesystem to store the keys directly, without managing serialization and file management manually:

```python
circuit.keys.save("/path/to/keys")
```

{% hint style="warning" %}
Keys are not saved in encrypted form. Please make sure you store them in a safe environment, or encrypt them manually after saving.
{% endhint %}

## Loading

After saving keys to disk, you can load them back using:

```python
circuit.keys.load("/path/to/keys")
```

## Automatic Management

If you want to generate keys in the first run and reuse the keys in consecutive runs, use:

```python
circuit.keys.load_if_exists_generate_and_save_otherwise("/path/to/keys")
```
