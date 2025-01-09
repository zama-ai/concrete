# Parameters compatibility with restrictions

This document explains how to use restrictions to limit the possible crypto-parameters used for the keys.

When compiling a module, the optimizer analyzes the circuits and the expected probability of error, to identify the fastest crypto-parameters that meet the specific constraints. The chosen crypto-parameters determine the size of the keys and the ciphertexts. This means that if an existing module is used in production with a specific set of crypto-parameters, there is no guarantee that a compilation of a second, different module will yield compatible crypto-parameters.

With _restrictions_, Concrete provides a way to ensure that a compilation generates compatible crypto-parameters. Restrictions will limit the search-space walked by the optimizer to ensure that only compatible parameters can be returned. As of now, we support two major restrictions:

+ [__Keyset restriction__](#keyset-restriction) : Restricts the crypto-parameters to an existing keyset.
+ [__Ranges restriction__](#ranges-restriction) : Restricts the crypto-parameters ranges allowed in the optimizer.

## Keyset restriction

You can generate keyset restriction directly form an existing keyset:

```python
@fhe.module()
class Big:
    @fhe.function({"x": "encrypted"})
    def inc(x):
        return (x + 1) % 200

big_inputset = [np.random.randint(1, 200, size=()) for _ in range(100)]
big_module = Big.compile(
    {"inc": big_inputset},
)
big_keyset_info = big_module.keys.specs.program_info.get_keyset_info()
big_module.keygen()

# We get the restriction from the existing keyset
restriction = big_keyset_info.get_restriction()

@fhe.module()
class Small:
    @fhe.function({"x": "encrypted"})
    def inc(x):
        return (x + 1) % 20

small_inputset = [np.random.randint(1, 20, size=()) for _ in range(100)]
small_module = Small.compile(
    {"inc": small_inputset},
    # We pass the keyset restriction as an extra compilation option
    keyset_restriction=restriction
)
restricted_keyset_info = restricted_module.keys.specs.program_info.get_keyset_info()
assert big_keyset_info == restricted_keyset_info

small_module.keys = big_module.keys

x = 5
x_enc = small_module.inc.encrypt(x)
```

## Ranges restriction

You can build a ranges restriction by adding available values:
```python
@fhe.module()
class Module:
    @fhe.function({"x": "encrypted"})
    def inc(x):
        return (x + 1) % 20

inputset = [np.random.randint(1, 20, size=()) for _ in range(100)]

## We generate a range restriction
range_restriction = RangeRestriction()

## Make 999 and 200 available as internal lwe dimensions
range_restriction.add_available_internal_lwe_dimension(999)
range_restriction.add_available_internal_lwe_dimension(200)

## Setting other restrictions
range_restriction.add_available_glwe_log_polynomial_size(12)
range_restriction.add_available_glwe_dimension(2)
range_restriction.add_available_pbs_level_count(3)
range_restriction.add_available_pbs_base_log(11)
range_restriction.add_available_ks_level_count(3)
range_restriction.add_available_ks_base_log(6)

module = Module.compile(
    {"inc": inputset},
    # We pass the range restriction as an extra compilation option.
    range_restriction=range_restriction
)
```

Note that if no available parameters are set for one of the parameter ranges (say `ks_base_log`), it is assumed that the default range is available.
