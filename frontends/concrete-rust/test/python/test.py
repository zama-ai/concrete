from concrete import fhe

@fhe.module()
class MyModule:

    @fhe.function({"x": "encrypted"})
    def inc(x):
        return (x + 1) % 20

    @fhe.function({"x": "encrypted"})
    def dec(x):
        return (x - 1) % 20

inputset = list(range(20))
my_module = MyModule.compile({"inc": inputset, "dec": inputset})
my_module.server.save("test.zip", via_mlir=True)
