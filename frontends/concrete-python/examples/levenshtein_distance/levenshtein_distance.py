# Computing Levenstein distance between strings, https://en.wikipedia.org/wiki/Levenshtein_distance

import time
from functools import lru_cache

import numpy

from concrete import fhe

# Parameters to be set by the user
max_string_length = 6


# Module FHE
@fhe.module()
class MyModule:
    @fhe.function({"x": "encrypted", "y": "encrypted"})
    def equal(x, y):
        return fhe.univariate(lambda x: x == 0)(x - y)

    @fhe.function(
        {
            "is_equal": "encrypted",
            "if_equal": "encrypted",
            "case_1": "encrypted",
            "case_2": "encrypted",
            "case_3": "encrypted",
        }
    )
    def mix(is_equal, if_equal, case_1, case_2, case_3):
        min_12 = numpy.minimum(case_1, case_2)
        min_123 = numpy.minimum(min_12, case_3)

        return fhe.if_then_else(is_equal, if_equal, 1 + min_123)

    # There is a single output in mix: it can go to
    #   - input 1 of mix
    #   - input 2 of mix
    #   - input 3 of mix
    #   - input 4 of mix
    # or just be the final output
    #
    # There is a single output of equal, it goes to input 0 of mix
    composition = fhe.Wired(
        [
            fhe.Wire(fhe.AllOutputs(equal), fhe.Input(mix, 0)),
            fhe.Wire(fhe.AllOutputs(mix), fhe.Input(mix, 1)),
            fhe.Wire(fhe.AllOutputs(mix), fhe.Input(mix, 2)),
            fhe.Wire(fhe.AllOutputs(mix), fhe.Input(mix, 3)),
            fhe.Wire(fhe.AllOutputs(mix), fhe.Input(mix, 4)),
        ]
    )


# For now, we pick only small letters
def random_letter_as_int():
    return 97 + numpy.random.randint(26)


def random_letter():
    return chr(random_letter_as_int())


def random_string(l):
    return "".join([random_letter() for _ in range(l)])


def map_string_to_int(s):
    return tuple(ord(si) for si in s)


# Compilation
inputset_equal = [(random_letter_as_int(), random_letter_as_int()) for _ in range(1000)]
inputset_mix = [
    (
        numpy.random.randint(2),
        numpy.random.randint(max_string_length),
        numpy.random.randint(max_string_length),
        numpy.random.randint(max_string_length),
        numpy.random.randint(max_string_length),
    )
    for _ in range(100)
]

my_module = MyModule.compile(
    {"equal": inputset_equal, "mix": inputset_mix},
    show_mlir=True,
    p_error=10**-20,
    show_optimizer=True,
)


# Function in clear, for reference and comparison
@lru_cache
def levenshtein_clear(x, y):
    if len(x) == 0:
        return len(y)
    if len(y) == 0:
        return len(x)

    if x[0] == y[0]:
        return levenshtein_clear(x[1:], y[1:])

    case_1 = levenshtein_clear(x[1:], y)
    case_2 = levenshtein_clear(x, y[1:])
    case_3 = levenshtein_clear(x[1:], y[1:])

    return 1 + min(case_1, case_2, case_3)


# Function in FHE-simulate, to debug
@lru_cache
def levenshtein_simulate(x, y):
    if len(x) == 0:
        return len(y)
    if len(y) == 0:
        return len(x)

    if_equal = levenshtein_simulate(x[1:], y[1:])
    case_1 = levenshtein_simulate(x[1:], y)
    case_2 = levenshtein_simulate(x, y[1:])
    case_3 = if_equal

    is_equal = my_module.equal(x[0], y[0])
    returned_value = my_module.mix(is_equal, if_equal, case_1, case_2, case_3)

    return returned_value


# Function in FHE
@lru_cache
def levenshtein_fhe(x, y):
    if len(x) == 0:
        # In clear, that's return len(y)
        return my_module.mix.encrypt(None, len(y), None, None, None)[1]
    if len(y) == 0:
        # In clear, that's return len(x)
        return my_module.mix.encrypt(None, len(x), None, None, None)[1]

    if_equal = levenshtein_fhe(x[1:], y[1:])
    case_1 = levenshtein_fhe(x[1:], y)
    case_2 = levenshtein_fhe(x, y[1:])
    case_3 = if_equal

    # In FHE
    is_equal = my_module.equal.run(x[0], y[0])
    returned_value = my_module.mix.run(is_equal, if_equal, case_1, case_2, case_3)

    return returned_value


# Random patterns of different lengths
list_patterns = [
    ("", ""),
    ("", "a"),
    ("b", ""),
    ("a", "a"),
    ("a", "b"),
]

for length_1 in range(max_string_length + 1):
    for length_2 in range(max_string_length + 1):
        list_patterns += [
            (
                random_string(length_1),
                random_string(length_2),
            )
            for _ in range(1)
        ]

# Checks in simulation
print("Computations in simulation\n")

for a, b in list_patterns:

    print(f"    Computing Levenshtein between strings '{a}' and '{b}'", end="")

    assert len(a) <= max_string_length
    assert len(b) <= max_string_length

    a_as_int = map_string_to_int(a)
    b_as_int = map_string_to_int(b)

    l1_simulate = levenshtein_simulate(a_as_int, b_as_int)
    l1_clear = levenshtein_clear(a_as_int, b_as_int)

    assert l1_simulate == l1_clear, f"    {l1_simulate=} and {l1_clear=} are different"
    print(" - OK")

# Key generation
my_module.keygen()

# Checks in FHE
print("\nComputations in FHE\n")

for a, b in list_patterns:

    print(f"    Computing Levenshtein between strings '{a}' and '{b}'", end="")

    assert len(a) <= max_string_length
    assert len(b) <= max_string_length

    a_as_int = map_string_to_int(a)
    b_as_int = map_string_to_int(b)

    a_enc = tuple(my_module.equal.encrypt(ai, None)[0] for ai in a_as_int)
    b_enc = tuple(my_module.equal.encrypt(None, bi)[1] for bi in b_as_int)

    time_begin = time.time()
    l1_fhe_enc = levenshtein_fhe(a_enc, b_enc)
    time_end = time.time()

    l1_fhe = my_module.mix.decrypt(l1_fhe_enc)

    l1_clear = levenshtein_clear(a, b)

    assert l1_fhe == l1_clear, f"    {l1_fhe=} and {l1_clear=} are different"
    print(f" - OK in {time_end - time_begin:.2f} seconds")
