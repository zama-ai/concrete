# Deploy

After developing your circuit, you may want to deploy it. However, sharing the details of your circuit with every client might not be desirable. You might want to perform the computation in dedicated servers, as well. In this case, you can use the `Client` and `Server` features of **Concrete**.

## Development of the circuit

You can develop your circuit like we've discussed in the previous chapters. Here is a simple example:

<!--pytest-codeblocks:skip-->
```python
from concrete import fhe

@fhe.compiler({"x": "encrypted"})
def function(x):
    return x + 42

inputset = range(10)
circuit = function.compile(inputset)
```

Once you have your circuit, you can save everything the server needs:

<!--pytest-codeblocks:skip-->
```python
circuit.server.save("server.zip")
```

Then, send `server.zip` to your computation server.

## Setting up a server

You can load the `server.zip` you get from the development machine:

<!--pytest-codeblocks:skip-->
```python
from concrete import fhe

server = fhe.Server.load("server.zip")
```

You will need to wait for requests from clients. The first likely request is for `ClientSpecs`.

Clients need `ClientSpecs` to generate keys and request computation. You can serialize `ClientSpecs`:

<!--pytest-codeblocks:skip-->
```python
serialized_client_specs: str = server.client_specs.serialize()
```

Then, you can send it to the clients requesting it.

## Setting up clients

After getting the serialized `ClientSpecs` from a server, you can create the client object:

<!--pytest-codeblocks:skip-->
```python
client_specs = fhe.ClientSpecs.deserialize(serialized_client_specs)
client = fhe.Client(client_specs)
```

## Generating keys (on the client)

Once you have the `Client` object, you can perform key generation:

<!--pytest-codeblocks:skip-->
```python
client.keys.generate()
```

This method generates encryption/decryption keys and evaluation keys.

The server requires evaluation keys linked to the encryption keys that you just generated. You can serialize your evaluation keys as shown:

<!--pytest-codeblocks:skip-->
```python
serialized_evaluation_keys: bytes = client.evaluation_keys.serialize()
```

After serialization, send the evaluation keys to the server.

{% hint style="info" %}
Serialized evaluation keys are very big in terms of size, so you may want to cache them on the server instead of sending them with each request.
{% endhint %}

## Encrypting inputs (on the client)

Now encrypt your inputs and request the server to perform the computation. You can do it like so:

<!--pytest-codeblocks:skip-->
```python
arg: fhe.Data = client.encrypt(7)
serialized_arg: bytes = arg.serialize()
```

Then, send serialized args to the server.

## Performing computation (on the server)

Once you have serialized evaluation keys and serialized arguments, you can deserialize them:

<!--pytest-codeblocks:skip-->
```python
deserialized_evaluation_keys = fhe.EvaluationKeys.deserialize(serialized_evaluation_keys)
deserialized_arg = fhe.Data.deserialize(serialized_arg)
```

You can perform the computation, as well:

<!--pytest-codeblocks:skip-->
```python
result: fhe.Data = server.run(deserialized_arg, evaluation_keys=deserialized_evaluation_keys)
serialized_result: bytes = result.serialize()
```

Then, send the serialized public result back to the client, so they can decrypt it and get the result of the computation.

## Decrypting the result (on the client)

Once you have received the public result of the computation from the server, you can deserialize it:

<!--pytest-codeblocks:skip-->
```python
deserialized_result = fhe.Data.deserialize(serialized_result)
```

Then, decrypt the result:

<!--pytest-codeblocks:skip-->
```python
decrypted_result = client.decrypt(deserialized_result)
assert decrypted_result == 49
```
