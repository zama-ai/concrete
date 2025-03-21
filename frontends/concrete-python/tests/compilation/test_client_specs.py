"""
Tests of `ClientSpecs` class.
"""

import json

from concrete.fhe.compilation import ClientSpecs
from concrete.fhe.compilation.server import ProgramInfo

PROGRAM_INFO_NO_TFHERS_SPECS = (
    b'{"keyset":{"lweSecretKeys":[{"id":0,"params":{"lweDimension":2048,"integerPr'
    b'ecision":64,"keyType":"binary"}},{"id":1,"params":{"lweDimension":4096,"inte'
    b'gerPrecision":64,"keyType":"binary"}},{"id":2,"params":{"lweDimension":890,"'
    b'integerPrecision":64,"keyType":"binary"}},{"id":3,"params":{"lweDimension":5'
    b'91,"integerPrecision":64,"keyType":"binary"}},{"id":4,"params":{"lweDimensio'
    b'n":2048,"integerPrecision":64,"keyType":"binary"}},{"id":5,"params":{"lweDim'
    b'ension":614,"integerPrecision":64,"keyType":"binary"}}],"lweBootstrapKeys":['
    b'{"id":0,"inputId":2,"outputId":0,"params":{"levelCount":1,"baseLog":23,"glwe'
    b'Dimension":1,"polynomialSize":2048,"variance":8.4422531129329586e-31,"intege'
    b'rPrecision":64,"modulus":{"modulus":{"native":{}}},"keyType":"binary","input'
    b'LweDimension":890},"compression":"none"},{"id":1,"inputId":3,"outputId":4,"p'
    b'arams":{"levelCount":3,"baseLog":12,"glweDimension":4,"polynomialSize":512,"'
    b'variance":8.4422531129329586e-31,"integerPrecision":64,"modulus":{"modulus":'
    b'{"native":{}}},"keyType":"binary","inputLweDimension":591},"compression":"no'
    b'ne"},{"id":2,"inputId":5,"outputId":4,"params":{"levelCount":2,"baseLog":16,'
    b'"glweDimension":4,"polynomialSize":512,"variance":8.4422531129329586e-31,"in'
    b'tegerPrecision":64,"modulus":{"modulus":{"native":{}}},"keyType":"binary","i'
    b'nputLweDimension":614},"compression":"none"}],"lweKeyswitchKeys":[{"id":0,"i'
    b'nputId":1,"outputId":2,"params":{"levelCount":3,"baseLog":5,"variance":6.948'
    b'3087989794833e-13,"integerPrecision":64,"modulus":{"modulus":{"native":{}}},'
    b'"keyType":"binary","inputLweDimension":4096,"outputLweDimension":890},"compr'
    b'ession":"none"},{"id":1,"inputId":0,"outputId":1,"params":{"levelCount":1,"b'
    b'aseLog":31,"variance":4.70197740328915e-38,"integerPrecision":64,"modulus":{'
    b'"modulus":{"native":{}}},"keyType":"binary","inputLweDimension":2048,"output'
    b'LweDimension":4096},"compression":"none"},{"id":2,"inputId":1,"outputId":3,"'
    b'params":{"levelCount":3,"baseLog":3,"variance":2.9356807113397357e-08,"integ'
    b'erPrecision":64,"modulus":{"modulus":{"native":{}}},"keyType":"binary","inpu'
    b'tLweDimension":4096,"outputLweDimension":591},"compression":"none"},{"id":3,'
    b'"inputId":4,"outputId":1,"params":{"levelCount":1,"baseLog":31,"variance":4.'
    b'70197740328915e-38,"integerPrecision":64,"modulus":{"modulus":{"native":{}}}'
    b',"keyType":"binary","inputLweDimension":2048,"outputLweDimension":4096},"com'
    b'pression":"none"},{"id":4,"inputId":4,"outputId":5,"params":{"levelCount":3,'
    b'"baseLog":3,"variance":1.2938256742325373e-08,"integerPrecision":64,"modulus'
    b'":{"modulus":{"native":{}}},"keyType":"binary","inputLweDimension":2048,"out'
    b'putLweDimension":614},"compression":"none"}],"packingKeyswitchKeys":[]},"cir'
    b'cuits":[{"inputs":[{"rawInfo":{"shape":{"dimensions":[5,4,4,4097]},"integerP'
    b'recision":64,"isSigned":false},"typeInfo":{"lweCiphertext":{"abstractShape":'
    b'{"dimensions":[5,4,4]},"concreteShape":{"dimensions":[5,4,4,4097]},"integerP'
    b'recision":64,"encryption":{"keyId":1,"variance":4.70197740328915e-38,"lweDim'
    b'ension":4096,"modulus":{"modulus":{"native":{}}}},"compression":"none","enco'
    b'ding":{"integer":{"width":5,"isSigned":false,"mode":{"native":{}}}}}}}],"out'
    b'puts":[{"rawInfo":{"shape":{"dimensions":[5,4,8,4097]},"integerPrecision":64'
    b',"isSigned":false},"typeInfo":{"lweCiphertext":{"abstractShape":{"dimensions'
    b'":[5,4,8]},"concreteShape":{"dimensions":[5,4,8,4097]},"integerPrecision":64'
    b',"encryption":{"keyId":1,"variance":4.70197740328915e-38,"lweDimension":4096'
    b',"modulus":{"modulus":{"native":{}}}},"compression":"none","encoding":{"inte'
    b'ger":{"width":5,"isSigned":false,"mode":{"native":{}}}}}}}],"name":"compute"'
    b"}]}"
)


def test_client_spec_without_tfhers_specs():
    """
    Test that a `ClientSpecs` instance can be created without `TFHERSClientSpecs`.

    This is for backward compatibility (before we introduced TFHERSClientSpecs).
    """
    program_info = ProgramInfo.deserialize(PROGRAM_INFO_NO_TFHERS_SPECS)
    client_specs = ClientSpecs(program_info=program_info)

    assert client_specs.program_info == program_info
    assert client_specs.tfhers_specs is None
    assert json.loads(client_specs.serialize()) == json.loads(PROGRAM_INFO_NO_TFHERS_SPECS)
    assert ClientSpecs.deserialize(PROGRAM_INFO_NO_TFHERS_SPECS) == client_specs
