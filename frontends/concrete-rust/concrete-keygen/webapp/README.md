# Chunked KeyGen WebApp

> [!IMPORTANT]
> This is a demo app that demonstrates how to use the chunked key generation. It's not meant for production.

### Build the Wasm module
Make sure you run `make build-webapp` in the parent directory first so that you have the wasm module under pkg/

### Serve the WebApp
You can serve the current directory with a simple HTTP server. You can run a dev server using python using our `server.py` (it avoids caching).

```bash
$ # from current directory
$ python3 server.py
```
### Create a Keyset Info file

In `concrete-python` you can do

```python
ks_info = circuit.client.specs.program_info.get_keyset_info()
serialized_key_info = keyset_info.serialize()
with open("ks_info.capnp", "wb") as f:
    f.write(serialized_key_info)
```

### Generate the Chunked Keyset

In the WebApp, upload your file, and choose a Chunk Size. The bigger it is the more memory the generation will need. Adapt it to your needs. You will get a Zip file at the end.

### Use the Chunked Keyset

You will need to assemble the Chunked Keyset into a valid Concrete Keyset. You can do that with the `keyasm` binary.

```bash
$ # in the parent folder
$ make build
$ ./target/release/keyasm <chunked_keyset_zip> <output_keyset>
```

You can then load this Keyset in `concrete-python`

```python
circuit.keys.load("<output_keyset>")
```
