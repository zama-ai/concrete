# TFHE-rs Interoperability

{% hint style="warning" %}

This feature is currently in beta version. Please note that the API may change in future Concrete releases.

{% endhint %}

This guide explains how to combine Concrete and [TFHE-rs](https://github.com/zama-ai/tfhe-rs) computations together. This allows you to convert ciphertexts from Concrete to TFHE-rs, and vice versa, and to run a computation with both libraries without requiring a decryption.

## Overview

There are differences between Concrete and TFHE-rs, so ensuring interoperability between them involves more than just data serialization. To achieve interoperability, we need to consider two main aspects.

#### Encoding differences

Both TFHE-rs and Concrete libraries use Learning with errors(LWE) ciphertexts, but integers are encoded differently:

- In Concrete, integers are simply encoded in a single ciphertext
- In TFHE-rs, integers are encoded into multiple ciphertext using radix decomposition

Converting between Concrete and TFHE-rs encrypted integers then require doing an encrypted conversion between the two different encodings.

When working with a TFHE-rs integer type in Concrete, you can use the `.encode(...)` and `.decode(...)` functions to see this in practice:

```python
from concrete.fhe import tfhers

# This will create a TFHE-rs unsigned integer of 8 bits
# using the parameters from the json file
tfhers_type = tfhers.get_type_from_params(
    "tfhers_params.json",
    is_signed=False,
    precision=8,
)

# Encoding could change depending on the parameters saved in 'tfhers_params.json'
# You should have the same result if message_modulus was equal to 4
assert (tfhers_type.encode(123) == [3, 2, 3, 1]).all()

assert tfhers_type.decode([3, 2, 3, 1]) == 123
```

#### Parameter match

The Concrete Optimizer may find parameters which are not in TFHE-rs's pre-computed list. To ensure interoperability, you need to either fix or constrain the search space in parts of the circuit where interoperability is required. This ensures that compatible parameters are used consistently.

## Scenarios

There are 2 different approaches to using Concrete and THFE-rs depending on the situation.

- [Scenario 1: Shared secret key](./shared-key.md): In this scenario, a single party aims to combine both Concrete and TFHE-rs in a computation. In this case, a shared secret key will be used, while different keysets will be held for Concrete and TFHE-rs.
- [Scenario 2: Pregenerated TFHE-rs keys](./pregenerated-key.md): This scenario involves two parties, each with a pre-established set of TFHE-rs keysets. The objective is to compute on encrypted TFHE-rs data using Concrete. In this case, there is no shared secret key. The party using Concrete will rely solely on TFHE-rs public keys and must optimize the parameters accordingly, while the party using TFHE-rs handles encryption, decryption, and computation.

## Serialization of ciphertexts and keys

Concrete already has its serilization functions (such as `tfhers_bridge.export_value`, `tfhers_bridge.import_value`, `tfhers_bridge.keygen_with_initial_keys`, `tfhers_bridge.serialize_input_secret_key`, and so on). However, when implementing a TFHE-rs computation in Rust, we must use a compatible serialization. Learn more in [Serialization of ciphertexts and keys](./serialization.md).
