# Parameters compatibility with restrictions

When compiling a module, the optimizer analyzes the circuits and the expected probability of error, to find the fastest crypto-parameters suiting those constraints. Depending on the crypto-parameters found, the size of the keys (and the ciphertexts) will differ. This means that if an existing module is used in production (using a certain set of crypto-parameters), there is no guarantee that a compilation of a second (different) module will yield compatible crypto-parameters.

Concrete provides a way to ensure that a compilation is going to yield compatible crypto-parameters, thanks to _restrictions_. Restrictions are going to restrict the search-space walked by the optimizer to ensure that only compatible parameters can be returnedyielded. As of now, we support two major restrictions:

+ __Keyset restriction__ : Restricts the crypto-parameters to an existing keyset. This restriction is suited for users that already have a module in production, and want to compile a compatible module.
+ __Ranges restriction__ : Restricts the crypto-parameters ranges allowed in the optimizer. This restriction is suited to users targetting a specific backend which does not support the breadth of parameters available on CPU.

## Keyset restriction

The keyset restriction can be generated directly form an existing keyset:

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
```

## Ranges restriction

A ranges restriction can be built by adding available values:
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
