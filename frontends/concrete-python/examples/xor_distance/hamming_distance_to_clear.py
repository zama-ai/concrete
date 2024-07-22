import argparse
import time

import numpy as np

from concrete import fhe

# Hamming weight computation
hw_table_values = [np.binary_repr(x).count("1") for x in range(2**8)]

# fmt: off
assert np.array_equal(hw_table_values, [
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3,
    4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4,
    4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2,
    3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5,
    4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3,
    4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3,
    3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5,
    6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6,
    4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5,
    6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8]
)
# fmt: on

hw = fhe.LookupTable(hw_table_values)


def mapme(x):
    """Map 0 to -1, and keep 1 as 1."""
    return 2 * x - 1


def dist_in_clear(x, y):
    """Compute the distance in the clear."""
    return np.sum(hw[x ^ y])


def dist_in_fhe(x_mapped, y_mapped):
    """Compute the distance in FHE."""

    # x is a line tensor, whose 0's have been replaced by -1
    # y_clear is a column tensor, whose 0's have been replaced by -1
    assert x_mapped.ndim == y_mapped.ndim == 2
    assert x_mapped.shape[0] == y_mapped.shape[1] == 1

    u = np.matmul(x_mapped, y_mapped)[0][0]

    # So, u is a scalar:
    # - bits which are the same between x and y_clear (either two -1's or two 1's) count for a +1
    #   in the scalar
    # - bits which are different between x and y_clear (either (-1, 1) or (1, -1)) count for a -1
    #   in the scalar
    # Hence the HW distance is (len(x) - u) / 2
    final_result = np.prod(x_mapped.shape) - u

    # The result which is returned is the double of the distance, we'll halve this in the clear
    return final_result


def manage_args():
    """Manage user args."""
    parser = argparse.ArgumentParser(
        description="Hamming weight (aka XOR) distance in Concrete, between an encrypted vector "
        "and a clear vector."
    )
    parser.add_argument(
        "--nb_bits",
        dest="nb_bits",
        action="store",
        type=int,
        default=120,
        help="Number of bits (better to be a multiple of 12 to test all bitwidths)",
    )
    parser.add_argument(
        "--show_mlir",
        dest="show_mlir",
        action="store_true",
        help="Show the MLIR",
    )
    parser.add_argument(
        "--repeat",
        dest="repeat",
        action="store",
        type=int,
        default=5,
        help="Repeat x times",
    )
    args = parser.parse_args()
    return args


def main():
    """Main function."""
    print()

    # Options by the user
    args = manage_args()

    nb_bits = args.nb_bits

    # Info
    print(
        f"Computing XOR distance on {nb_bits} bits using algorithm dist_in_fhe, using vectors of "
        "1b cells"
    )

    # Compile the circuit
    inputset = [
        (
            mapme(np.random.randint(2**1, size=(1, nb_bits))),
            mapme(np.transpose(np.random.randint(2**1, size=(1, nb_bits)))),
        )
        for _ in range(100)
    ]

    compiler = fhe.Compiler(dist_in_fhe, {"x_mapped": "encrypted", "y_mapped": "clear"})
    circuit = compiler.compile(
        inputset,
        show_mlir=args.show_mlir,
        bitwise_strategy_preference=fhe.BitwiseStrategy.ONE_TLU_PROMOTED,
        multivariate_strategy_preference=fhe.MultivariateStrategy.PROMOTED,
    )

    # Then generate the keys
    circuit.keygen()

    total_time = 0

    nb_samples_for_warmup = 10

    # Then use
    for i in range(nb_samples_for_warmup + args.repeat):
        # Take a random input pair
        x, y = (
            np.random.randint(2**1, size=(1, nb_bits)),
            np.random.randint(2**1, size=(1, nb_bits)),
        )

        x_mapped = mapme(x)
        y_mapped = mapme(np.transpose(y))

        # Encrypt
        encrypted_input = circuit.encrypt(x_mapped, y_mapped)

        # Compute the distance in FHE
        begin_time = time.time()
        encrypted_result = circuit.run(encrypted_input)
        end_time = time.time()

        # Don't count the warmup samples
        if i >= nb_samples_for_warmup:
            total_time += end_time - begin_time

        # Decrypt
        result = circuit.decrypt(encrypted_result)

        # Halve this in the clear, to have the final result
        result /= 2

        # Check
        assert result == dist_in_clear(x, y)

    average_time = total_time / args.repeat
    print(f"Distance between encrypted vectors done in {average_time:.2f} " f"seconds in average")


if __name__ == "__main__":
    main()
