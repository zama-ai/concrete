from concrete import fhe
from concrete.fhe import tfhers

tfhers_params = tfhers.CryptoParams(
    909,
    1,
    4096,
    15,
    2,
    0,
    2.168404344971009e-19,
    tfhers.EncryptionKeyChoice.BIG,
)
tfhers_dtype = tfhers.uint16_2_2(tfhers_params)

@fhe.module()
class MyModule:

    @fhe.function({"x": "encrypted", "y":"encrypted"})
    def my_func(x, y):
        x = tfhers.to_native(x)
        y = tfhers.to_native(y)
        return tfhers.from_native(x + y, tfhers_dtype)

def t(v):
    return tfhers.TFHERSInteger(tfhers_dtype, v)

inputset = [(t(0), t(0)), (t(2**14), t(2**14))]
my_module = MyModule.compile({"my_func": inputset})
my_module.server.save("test_tfhers.zip", via_mlir=True)
