"""
Parameter.
"""

# pylint: disable=no-name-in-module,import-error

from typing import Union

from mlir._mlir_libs._concretelang._compiler import (
    LweSecretKeyParam,
    BootstrapKeyParam,
    KeyswitchKeyParam,
    PackingKeyswitchKeyParam,
    KeyType,
)

from .client_parameters import ClientParameters

# pylint: enable=no-name-in-module,import-error


class Parameter:
    """
    An FHE parameter.
    """

    _inner: Union[
        LweSecretKeyParam,
        BootstrapKeyParam,
        KeyswitchKeyParam,
        PackingKeyswitchKeyParam,
    ]

    def __init__(
        self,
        client_parameters: ClientParameters,
        key_type: KeyType,
        key_index: int,
    ):
        if key_type == KeyType.SECRET:
            self._inner = client_parameters.cpp().secret_keys[key_index]
        elif key_type == KeyType.BOOTSTRAP:
            self._inner = client_parameters.cpp().bootstrap_keys[key_index]
        elif key_type == KeyType.KEY_SWITCH:
            self._inner = client_parameters.cpp().keyswitch_keys[key_index]
        elif key_type == KeyType.PACKING_KEY_SWITCH:
            self._inner = client_parameters.cpp().packing_keyswitch_keys[key_index]
        else:
            raise ValueError("invalid key type")

    def __getattr__(self, item):
        return getattr(self._inner, item)

    def __repr__(self):
        param = self._inner

        if isinstance(param, LweSecretKeyParam):
            result = f"LweSecretKeyParam(" f"dimension={param.dimension()}" f")"

        elif isinstance(param, BootstrapKeyParam):
            result = (
                f"BootstrapKeyParam("
                f"polynomial_size={param.polynomial_size()}, "
                f"glwe_dimension={param.glwe_dimension()}, "
                f"input_lwe_dimension={param.input_lwe_dimension()}, "
                f"level={param.level()}, "
                f"base_log={param.base_log()}, "
                f"variance={param.variance()}"
                f")"
            )

        elif isinstance(param, KeyswitchKeyParam):
            result = (
                f"KeyswitchKeyParam("
                f"level={param.level()}, "
                f"base_log={param.base_log()}, "
                f"variance={param.variance()}"
                f")"
            )

        elif isinstance(param, PackingKeyswitchKeyParam):
            result = (
                f"PackingKeyswitchKeyParam("
                f"polynomial_size={param.polynomial_size()}, "
                f"glwe_dimension={param.glwe_dimension()}, "
                f"input_lwe_dimension={param.input_lwe_dimension()}"
                f"level={param.level()}, "
                f"base_log={param.base_log()}, "
                f"variance={param.variance()}"
                f")"
            )

        else:
            assert False

        return result

    def __str__(self):
        return repr(self)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)
