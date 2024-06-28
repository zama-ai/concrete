# Computing Levenstein distance between strings, https://en.wikipedia.org/wiki/Levenshtein_distance

import time
import argparse
import random
from functools import lru_cache

import numpy

from concrete import fhe

# Module FHE
@fhe.module()
class MyModule:
    @fhe.function({"x": "encrypted", "y": "encrypted"})
    def equal(x, y):
        return x == y

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


def random_pick_in_values(mapping_to_int):
    return numpy.random.randint(len(mapping_to_int))


def random_pick_in_keys(mapping_to_int):
    return random.choice(list(mapping_to_int))


def random_string(mapping_to_int, l):
    return "".join([random_pick_in_keys(mapping_to_int) for _ in range(l)])


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
def levenshtein_simulate(my_module, x, y):
    if len(x) == 0:
        return len(y)
    if len(y) == 0:
        return len(x)

    if_equal = levenshtein_simulate(my_module, x[1:], y[1:])
    case_1 = levenshtein_simulate(my_module, x[1:], y)
    case_2 = levenshtein_simulate(my_module, x, y[1:])
    case_3 = if_equal

    is_equal = my_module.equal(x[0], y[0])
    returned_value = my_module.mix(is_equal, if_equal, case_1, case_2, case_3)

    return returned_value


# Function in FHE
@lru_cache
def levenshtein_fhe(my_module, x, y):
    if len(x) == 0:
        # In clear, that's return len(y)
        return my_module.mix.encrypt(None, len(y), None, None, None)[1]
    if len(y) == 0:
        # In clear, that's return len(x)
        return my_module.mix.encrypt(None, len(x), None, None, None)[1]

    if_equal = levenshtein_fhe(my_module, x[1:], y[1:])
    case_1 = levenshtein_fhe(my_module, x[1:], y)
    case_2 = levenshtein_fhe(my_module, x, y[1:])
    case_3 = if_equal

    # In FHE
    is_equal = my_module.equal.run(x[0], y[0])
    returned_value = my_module.mix.run(is_equal, if_equal, case_1, case_2, case_3)

    return returned_value


# Manage user args
def manage_args():
    parser = argparse.ArgumentParser(description="Levenshtein distance in Concrete.")
    parser.add_argument(
        "--show_mlir",
        dest="show_mlir",
        action="store_true",
        help="Show the MLIR",
    )
    parser.add_argument(
        "--show_optimizer",
        dest="show_optimizer",
        action="store_true",
        help="Show the optimizer outputs",
    )
    parser.add_argument(
        "--autotest",
        dest="autotest",
        action="store_true",
        help="Run random tests",
    )
    parser.add_argument(
        "--autoperf",
        dest="autoperf",
        action="store_true",
        help="Run benchmarks",
    )
    parser.add_argument(
        "--alphabet",
        dest="alphabet",
        choices=["string", "STRING", "StRiNg", "ACTG"],
        default="string",
        help="Setting the alphabet",
    )
    parser.add_argument(
        "--max_string_length",
        dest="max_string_length",
        type=int,
        default=4,
        help="Setting the maximal size of strings",
    )
    args = parser.parse_args()
    return args


def compile_module(mapping_to_int, args):
    # Compilation
    inputset_equal = [
        (random_pick_in_values(mapping_to_int), random_pick_in_values(mapping_to_int))
        for _ in range(1000)
    ]
    inputset_mix = [
        (
            numpy.random.randint(2),
            numpy.random.randint(args.max_string_length),
            numpy.random.randint(args.max_string_length),
            numpy.random.randint(args.max_string_length),
            numpy.random.randint(args.max_string_length),
        )
        for _ in range(100)
    ]

    my_module = MyModule.compile(
        {"equal": inputset_equal, "mix": inputset_mix},
        show_mlir=args.show_mlir,
        p_error=10**-20,
        show_optimizer=args.show_optimizer,
        comparison_strategy_preference=fhe.ComparisonStrategy.ONE_TLU_PROMOTED,
        min_max_strategy_preference=fhe.MinMaxStrategy.ONE_TLU_PROMOTED,
    )

    return my_module


def prepare_alphabet_mapping(alphabet, verbose=True):
    if alphabet == "string":
        letters = "".join([chr(97 + i) for i in range(26)])
    elif alphabet == "STRING":
        letters = "".join([chr(65 + i) for i in range(26)])
    elif alphabet == "StRiNg":
        letters = "".join([chr(97 + i) for i in range(26)] + [chr(65 + i) for i in range(26)])
    elif alphabet == "ACTG":
        letters = "ACTG"
    else:
        raise ValueError(f"Unknown alphabet {alphabet}")

    if verbose:
        print(f"Making random tests with alphabet {alphabet}")
        print(f"Letters are {letters}\n")

    mapping_to_int = {}

    for i, c in enumerate(letters):
        mapping_to_int[c] = i

    return mapping_to_int


def prepare_random_patterns(mapping_to_int, len_min, len_max, nb_strings):
    # Random patterns of different lengths
    list_patterns = []
    for _ in range(nb_strings):
        for length_1 in range(len_min, len_max + 1):
            for length_2 in range(len_min, len_max + 1):
                list_patterns += [
                    (
                        random_string(mapping_to_int, length_1),
                        random_string(mapping_to_int, length_2),
                    )
                    for _ in range(1)
                ]

    return list_patterns


def compute_in_simulation(my_module, list_patterns, mapping_to_int):

    # Checks in simulation
    print("Computations in simulation\n")

    for a, b in list_patterns:

        print(f"    Computing Levenshtein between strings '{a}' and '{b}'", end="")

        a_as_int = tuple([mapping_to_int[ai] for ai in a])
        b_as_int = tuple([mapping_to_int[bi] for bi in b])

        l1_simulate = levenshtein_simulate(my_module, a_as_int, b_as_int)
        l1_clear = levenshtein_clear(a_as_int, b_as_int)

        assert l1_simulate == l1_clear, f"    {l1_simulate=} and {l1_clear=} are different"
        print(" - OK")


def compute_in_fhe(my_module, list_patterns, mapping_to_int, verbose=False):
    # Key generation
    my_module.keygen()

    # Checks in FHE
    if verbose:
        print("\nComputations in FHE\n")

    for a, b in list_patterns:

        print(f"    Computing Levenshtein between strings '{a}' and '{b}'", end="")

        a_as_int = [mapping_to_int[ai] for ai in a]
        b_as_int = [mapping_to_int[bi] for bi in b]

        a_enc = tuple(my_module.equal.encrypt(ai, None)[0] for ai in a_as_int)
        b_enc = tuple(my_module.equal.encrypt(None, bi)[1] for bi in b_as_int)

        time_begin = time.time()
        l1_fhe_enc = levenshtein_fhe(my_module, a_enc, b_enc)
        time_end = time.time()

        l1_fhe = my_module.mix.decrypt(l1_fhe_enc)

        l1_clear = levenshtein_clear(a, b)

        assert l1_fhe == l1_clear, f"    {l1_fhe=} and {l1_clear=} are different"
        print(f" - OK in {time_end - time_begin:.2f} seconds")


def main():
    print()

    # Options by the user
    args = manage_args()

    # Do what the user requested
    if args.autotest:
        mapping_to_int = prepare_alphabet_mapping(args.alphabet)
        my_module = compile_module(mapping_to_int, args)
        list_patterns = prepare_random_patterns(mapping_to_int, 0, args.max_string_length, 1)
        compute_in_simulation(my_module, list_patterns, mapping_to_int)
        compute_in_fhe(my_module, list_patterns, mapping_to_int)
        print("")

    if args.autoperf:
        for alphabet in ["ACTG", "string", "STRING", "StRiNg"]:
            print(f"Typical performances for alphabet {alphabet}, with string of maximal length:\n")
            mapping_to_int = prepare_alphabet_mapping(alphabet, verbose=False)
            my_module = compile_module(mapping_to_int, args)
            list_patterns = prepare_random_patterns(
                mapping_to_int, args.max_string_length, args.max_string_length, 3
            )
            compute_in_fhe(my_module, list_patterns, mapping_to_int, verbose=False)
            print("")

    print("Successful end\n")


if __name__ == "__main__":
    main()
