from concrete import fhe
from concrete.fhe import tfhers

TFHERS_UINT_8_3_2_4096 = tfhers.TFHERSIntegerType(
    False,
    bit_width=8,
    carry_width=3,
    msg_width=2,
    params=tfhers.CryptoParams(
        lwe_dimension=909,
        glwe_dimension=1,
        polynomial_size=4096,
        pbs_base_log=15,
        pbs_level=2,
        lwe_noise_distribution=0,
        glwe_noise_distribution=2.168404344971009e-19,
        encryption_key_choice=tfhers.EncryptionKeyChoice.BIG,
    ),
)

@fhe.module()
class MyModule:

    @fhe.function({"x": "encrypted", "y":"encrypted"})
    def my_func(x, y):
        x = tfhers.to_native(x)
        y = tfhers.to_native(y)
        return tfhers.from_native(x + y, TFHERS_UINT_8_3_2_4096)

def t(v):
    return tfhers.TFHERSInteger(TFHERS_UINT_8_3_2_4096, v)

inputset = [(t(0), t(0)), (t(2**6), t(2**6))]
my_module = MyModule.compile({"my_func": inputset})
my_module.server.save("test_tfhers.zip", via_mlir=True)
