# Deploy

After developing your circuit, you may want to deploy it. However, sharing the details of your circuit with every client might not be desirable. Further, you might want to perform the computation in dedicated servers. In this case, you can use the `Client` and `Server` features of **Concrete-Numpy**.

## Development of the circuit

You can develop your circuit like we've discussed in the previous chapters. Here is a simple example:

<!--pytest-codeblocks:skip-->
```python
import concrete.numpy as cnp

@cnp.compiler({"x": "encrypted"})
def function(x):
    return x + 42

inputset = range(10)
circuit = function.compile(inputset)
```

Once you have your circuit, you can save everything the server needs like so:

<!--pytest-codeblocks:skip-->
```python
circuit.server.save("server.zip")
```

All you need to do now is to send `server.zip` to your computation server.

## Setting up a server

You can load the `server.zip` you get from the development machine as follows:

<!--pytest-codeblocks:skip-->
```python
import concrete.numpy as cnp

server = cnp.Server.load("server.zip")
```

At this point, you will need to wait for requests from clients. The first likely request is for `ClientSpecs`.

Clients need `ClientSpecs` to generate keys and request computation. You can serialize `ClientSpecs` like so:

<!--pytest-codeblocks:skip-->
```python
serialized_client_specs: str = server.client_specs.serialize()
```

Then, you can send it to the clients requesting it.

## Setting up clients

After getting the serialized `ClientSpecs` from a server, you can create the client object like this:

<!--pytest-codeblocks:skip-->
```python
client_specs = cnp.ClientSpecs.unserialize(serialized_client_specs)
client = cnp.Client(client_specs)
```

## Generating keys (on the client)

Once you have the `Client` object, you can perform key generation:

<!--pytest-codeblocks:skip-->
```python
client.keygen()
```

This method generates encryption/decryption keys and evaluation keys.

The server requires evaluation keys linked to the encryption keys that you just generated. You can serialize your evaluation keys as shown below:

<!--pytest-codeblocks:skip-->
```python
serialized_evaluation_keys: bytes = client.evaluation_keys.serialize()
```

After serialization, you can send the evaluation keys to the server.

{% hint style="info" %}
Serialized evaluation keys are very big in size, so you may want to cache them on the server instead of sending them with each request.
{% endhint %}

## Encrypting inputs (on the client)

You are now ready to encrypt your inputs and request the server to perform the computation. You can do it like so:

<!--pytest-codeblocks:skip-->
```python
serialized_args: bytes = client.encrypt(7).serialize()
```

The only thing left to do is to send serialized args to the server.

## Performing computation (on the server)

Upon having the serialized evaluation keys and serialized arguments, you can unserialize them like so:

<!--pytest-codeblocks:skip-->
```python
unserialized_evaluation_keys = cnp.EvaluationKeys.unserialize(serialized_evaluation_keys)
unserialized_args  = server.client_specs.unserialize_public_args(serialized_args)
```

And you can perform the computation as well:

<!--pytest-codeblocks:skip-->
```python
public_result = server.run(unserialized_args, unserialized_evaluation_keys)
serialized_public_result: bytes = public_result.serialize()
```

Finally, you can send the serialized public result back to the client, so they can decrypt it and get the result of the computation.

## Decrypting the result (on the client)

Once you have received the public result of the computation from the server, you can unserialize it:

<!--pytest-codeblocks:skip-->
```python
unserialized_public_result = client.specs.unserialize_public_result(serialized_public_result)
```

Finally, you can decrypt the result like so:

<!--pytest-codeblocks:skip-->
```python
result = client.decrypt(unserialized_public_result)
assert result == 49
```
