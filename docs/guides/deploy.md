# Deploy

After developing your circuit, you may want to deploy it. However, sharing the details of your circuit with every client might not be desirable. As well as this, you might want to perform the computation on dedicated servers. In this case, you can use the `Client` and `Server` features of **Concrete**.

## Development of the circuit

You can develop your circuit using the techniques discussed in previous chapters. Here is a simple example:

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

The server needs access to the evaluation keys that were just generated. You can serialize your evaluation keys as shown:

<!--pytest-codeblocks:skip-->
```python
serialized_evaluation_keys: bytes = client.evaluation_keys.serialize()
```

After serialization, send the evaluation keys to the server.

{% hint style="info" %}
Serialized evaluation keys are very large, so you may want to cache them on the server instead of sending them with each request.
{% endhint %}

## Encrypting inputs (on the client)

The next step is to encrypt your inputs and request the server to perform some computation. This can be done in the following way:

<!--pytest-codeblocks:skip-->
```python
arg: fhe.Value = client.encrypt(7)
serialized_arg: bytes = arg.serialize()
```

Then, send the serialized arguments to the server.

## Performing computation (on the server)

Once you have serialized evaluation keys and serialized arguments, you can deserialize them:

<!--pytest-codeblocks:skip-->
```python
deserialized_evaluation_keys = fhe.EvaluationKeys.deserialize(serialized_evaluation_keys)
deserialized_arg = fhe.Value.deserialize(serialized_arg)
```

You can perform the computation, as well:

<!--pytest-codeblocks:skip-->
```python
result: fhe.Value = server.run(deserialized_arg, evaluation_keys=deserialized_evaluation_keys)
serialized_result: bytes = result.serialize()
```

Then, send the serialized result back to the client. After this, the client can decrypt to receive the result of the computation.

## Decrypting the result (on the client)

Once you have received the serialized result of the computation from the server, you can deserialize it:

<!--pytest-codeblocks:skip-->
```python
deserialized_result = fhe.Value.deserialize(serialized_result)
```

Then, decrypt the result:

<!--pytest-codeblocks:skip-->
```python
decrypted_result = client.decrypt(deserialized_result)
assert decrypted_result == 49
```

# Deployment of modules

Deploying a [module](../compilation/composing_functions_with_modules.md) follows the same logic as the deployment of circuits. Assuming a module compiled in the following way:

<!--pytest-codeblocks:skip-->
```python
from concrete import fhe

@fhe.module()
class MyModule:
    @fhe.function({"x": "encrypted"})
    def inc(x):
        return x + 1

    @fhe.function({"x": "encrypted"})
    def dec(x):
        return x - 1

inputset = list(range(20))
my_module = MyModule.compile({"inc": inputset, "dec": inputset})
)
```

You can extract the server from the module and save it in a file:

<!--pytest-codeblocks:skip-->
```python
my_module.server.save("server.zip")
```

The only noticeable difference between the deployment of modules and the deployment of circuits is that the methods `Client::encrypt`, `Client::decrypt` and `Server::run` must contain an extra `function_name` argument specifying the name of the targeted function.

The encryption of an argument for the `inc` function of the module would be:

<!--pytest-codeblocks:skip-->
```python
arg = client.encrypt(7, function_name="inc")
serialized_arg = arg.serialize()
```

The execution of the `inc` function would be :

<!--pytest-codeblocks:skip-->
```python
result = server.run(deserialized_arg, evaluation_keys=deserialized_evaluation_keys, function_name="inc")
serialized_result = result.serialize()
```

Finally, decrypting a result from the execution of `dec` would be:

<!--pytest-codeblocks:skip-->
```python
decrypted_result = client.decrypt(deserialized_result, function_name="dec")
```
