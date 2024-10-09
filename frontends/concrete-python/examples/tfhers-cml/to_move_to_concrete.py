
import json
import math
from concrete.fhe import tfhers
from functools import partial

# FIXME: should we move this to Concrete library directly, hidden to the user
def get_tfhers_params_and_type_and_int(crypto_params_json, precision):

    def int_log2(x):
        return int(math.log2(x))

    # Read crypto parameters from TFHE-rs in the json file
    with open(crypto_params_json) as f:
        tfhe_rs_crypto_param_dic = json.load(f)

    # pretty_json = json.dumps(tfhe_rs_crypto_param_dic, indent=4)
    # print(pretty_json)

    def int_log2(x):
        return int(math.log2(x))

    # FIXME Params: users shouldn't change them, should we hide it
    pbs_tfhe_rs_crypto_param_dic = tfhe_rs_crypto_param_dic["inner"]["block_parameters"]["PBS"]
    lwe_dim = pbs_tfhe_rs_crypto_param_dic["lwe_dimension"]
    glwe_dim = pbs_tfhe_rs_crypto_param_dic["glwe_dimension"]
    poly_size = pbs_tfhe_rs_crypto_param_dic["polynomial_size"]
    pbs_base_log = pbs_tfhe_rs_crypto_param_dic["pbs_base_log"]
    pbs_level = pbs_tfhe_rs_crypto_param_dic["pbs_level"]
    msg_width = int_log2(pbs_tfhe_rs_crypto_param_dic["message_modulus"])
    carry_width = int_log2(pbs_tfhe_rs_crypto_param_dic["carry_modulus"])
    encryption_key_choice = tfhers.EncryptionKeyChoice.BIG
    lwe_noise_distr = pbs_tfhe_rs_crypto_param_dic["lwe_noise_distribution"]["Gaussian"]["std"]
    glwe_noise_distr = pbs_tfhe_rs_crypto_param_dic["glwe_noise_distribution"]["Gaussian"]["std"]

    assert glwe_dim == 1, "glwe dim must be 1"

    tfhers_params = tfhers.CryptoParams(
        lwe_dimension=lwe_dim,
        glwe_dimension=glwe_dim,
        polynomial_size=poly_size,
        pbs_base_log=pbs_base_log,
        pbs_level=pbs_level,
        lwe_noise_distribution=lwe_noise_distr,
        glwe_noise_distribution=glwe_noise_distr,
        encryption_key_choice=encryption_key_choice,
    )
    tfhers_type = tfhers.TFHERSIntegerType(
        is_signed=False,
        bit_width=precision,
        carry_width=carry_width,
        msg_width=msg_width,
        params=tfhers_params,
    )
    tfhers_int = partial(tfhers.TFHERSInteger, tfhers_type)

    return tfhers_params, tfhers_type, tfhers_int
