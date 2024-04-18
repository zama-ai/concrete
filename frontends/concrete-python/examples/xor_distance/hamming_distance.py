import argparse
import time

import numpy as np

from concrete import fhe


# Hamming weight computation
def hw(x):
    # Hamming Weight table for 8b entries
    hw_table_ref = [np.binary_repr(x).count("1") for x in range(2**8)]

    # fmt: off
    assert np.array_equal(hw_table_ref, [
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3,
        4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4,
        4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2,
        3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5,
        4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3,
        4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3,
        3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5,
        6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6,
        4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5,
        6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8])
    # fmt: on

    hw_table = fhe.LookupTable(hw_table_ref)
    return hw_table[x]


# Reference function for tests
def dist_in_clear(x, y):
    return np.sum(hw(x ^ y))


# For all bitsize_w
def dist_in_fhe_directly_from_cp(x, y):
    return np.sum(hw(x ^ y))


# For bitsize_w == 1 only
def dist_in_fhe_with_bits_1b(x, y):
    z = x + y
    zx = fhe.bits(z)[0]
    return np.sum(zx)


# For all bitsize_w
def dist_in_fhe_with_xor_internal(x, y, bitsize_w):
    power = 2**bitsize_w
    table = fhe.LookupTable([hw((i % power) ^ (i // power)) for i in range(power**2)])

    z = x + power * y
    zx = table[z]

    return np.sum(zx)


# For all bitsize_w
def dist_in_fhe_with_multivariate_internal(x, y):
    zx = fhe.multivariate(lambda x, y: hw(x ^ y))(x, y)
    return np.sum(zx)


# Manage user args
def manage_args():
    parser = argparse.ArgumentParser(description="Hamming weight (aka XOR) distance in Concrete.")
    parser.add_argument(
        "--nb_bits",
        dest="nb_bits",
        action="store",
        type=int,
        default=120,
        help="Number of bits (better to be a multiple of 12 to test all bitwidths)",
    )
    parser.add_argument(
        "--shape",
        dest="shape",
        action="store",
        type=int,
        nargs="+",
        default=None,
        help="How to shape the bits. It has almost no importance for speed",
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
    print()

    # Options by the user
    args = manage_args()

    nb_bits = args.nb_bits
    execution_times = {}

    for bitsize_w in [1, 2, 3, 4]:
        for algo in [
            "dist_in_fhe_with_bits_1b",
            "dist_in_fhe_with_xor_tables",
            "dist_in_fhe_with_multivariate_tables",
            "dist_in_fhe_directly_from_cp",
        ]:
            if algo == "dist_in_fhe_with_bits_1b":
                dist_function = dist_in_fhe_with_bits_1b
            elif algo == "dist_in_fhe_with_xor_tables":
                if bitsize_w == 1:
                    dist_function = lambda x, y: dist_in_fhe_with_xor_internal(x, y, 1)
                elif bitsize_w == 2:
                    dist_function = lambda x, y: dist_in_fhe_with_xor_internal(x, y, 2)
                elif bitsize_w == 3:
                    dist_function = lambda x, y: dist_in_fhe_with_xor_internal(x, y, 3)
                elif bitsize_w == 4:
                    dist_function = lambda x, y: dist_in_fhe_with_xor_internal(x, y, 4)
            elif algo == "dist_in_fhe_with_multivariate_tables":
                dist_function = dist_in_fhe_with_multivariate_internal
            else:
                assert algo == "dist_in_fhe_directly_from_cp"
                dist_function = dist_in_fhe_directly_from_cp

            if algo == "dist_in_fhe_with_bits_1b" and bitsize_w != 1:
                # Only work for 1b
                continue

            shape = (1, nb_bits // bitsize_w) if args.shape is None else tuple(args.shape)

            # Checks
            if nb_bits % bitsize_w != 0:
                print(
                    f"Number of bits is not a multiple of w, can't test this "
                    f"configuration {algo} {bitsize_w}"
                )
                continue

            assert (
                np.prod(shape) * bitsize_w == nb_bits
            ), "Your (shape, w) does not correspond to number of bits"

            # Info
            print(
                f"Computing XOR distance on {nb_bits} bits using algorithm {algo}, using vectors "
                f"of shape {shape} of {bitsize_w}b cells"
            )

            # Compile the circuit
            inputset = [
                (
                    np.random.randint(2**bitsize_w, size=shape),
                    np.random.randint(2**bitsize_w, size=shape),
                )
                for _ in range(100)
            ]

            compiler = fhe.Compiler(dist_function, {"x": "encrypted", "y": "encrypted"})
            circuit = compiler.compile(
                inputset,
                show_mlir=args.show_mlir,
                bitwise_strategy_preference=fhe.BitwiseStrategy.ONE_TLU_PROMOTED,
                multivariate_strategy_preference=fhe.MultivariateStrategy.PROMOTED,
            )

            # Then generate the keys
            circuit.keygen()

            total_time = 0

            # Then use
            for _i in range(args.repeat):
                # Take a random input pair
                x, y = (
                    np.random.randint(2**bitsize_w, size=shape),
                    np.random.randint(2**bitsize_w, size=shape),
                )

                # Encrypt
                encrypted_input = circuit.encrypt(x, y)

                # Compute the distance in FHE
                begin_time = time.time()
                encrypted_result = circuit.run(encrypted_input)
                end_time = time.time()

                total_time += end_time - begin_time

                # Decrypt
                result = circuit.decrypt(encrypted_result)

                # Check
                assert result == dist_in_clear(x, y)

            average_time = total_time / args.repeat
            print(
                f"Distance between encrypted vectors done in {average_time:.2f} "
                f"seconds in average"
            )

            execution_times[f"{algo} on {bitsize_w} bits"] = average_time

            print("")

    # Final results
    print("Results from the fastest to the slowest\n")
    sorted_execution_times = sorted(execution_times.items(), key=lambda x: x[1])

    for algo, average_time in sorted_execution_times:
        print(f"{algo:>50s}: {average_time:5.2f} seconds")

    print()


if __name__ == "__main__":
    main()
