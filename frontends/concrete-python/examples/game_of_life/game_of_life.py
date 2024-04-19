import sys
import time

# Hide pygame prompt
from os import environ

import numpy as np

environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

# ruff: noqa:E402
# pylint: disable=wrong-import-position,no-member
import argparse

# ruff: noqa:E402
import pygame

# ruff: noqa:E402
from concrete import fhe


# Function to workaround the miss of padding in CP
def by_hand_padding(original_grid, res):
    padded_res = fhe.zeros(original_grid.shape)

    original_grid_shape = original_grid.shape
    assert padded_res.shape[0:2] == (1, 1)
    padded_res[0, 0, 1 : original_grid_shape[2] - 1, 1 : original_grid_shape[3] - 1] = res

    assert original_grid.shape == padded_res.shape

    return padded_res


# Function to workaround the miss of padding in CP
def conv_with_hand_padding(grid, weight, do_padded_fix):
    convoluted_grid = fhe.conv(
        grid,
        weight.reshape(1, 1, *weight.shape),
        strides=(1, 1),
        dilations=(1, 1),
        group=1,
        pads=(1, 1, 1, 1) if not do_padded_fix else (0, 0, 0, 0),
    )

    if do_padded_fix:
        convoluted_grid = by_hand_padding(grid, convoluted_grid)

    return convoluted_grid


# Function for Game of Life
@fhe.compiler({"grid": "encrypted"})
def update_grid_method_3b(grid):
    # Method which uses two first TLU of 3 bits and a third TLU of 2 bits

    weights_method_3b_a = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ]
    )
    weights_method_3b_b = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
        ]
    )
    table_next_cell_3b_a = [i if i <= 3 else 4 for i in range(8)]
    table_next_cell_3b_b = [i - 1 if i in [2, 3] else 0 for i in range(6)]
    table_next_cell_3b_c = [int(i in [2, 3]) for i in range(4)]

    table_cp_next_cell_3b_a = fhe.LookupTable(table_next_cell_3b_a)
    table_cp_next_cell_3b_b = fhe.LookupTable(table_next_cell_3b_b)
    table_cp_next_cell_3b_c = fhe.LookupTable(table_next_cell_3b_c)

    # This is to workaround the fact that we have no pad option in fhe.conv
    do_padded_fix = True

    # Compute the sum of 7 elements
    convoluted_grid = conv_with_hand_padding(grid, weights_method_3b_a, do_padded_fix)

    # Apply a TLU: input in [0, 7], output in [0, 4]
    grid_a = table_cp_next_cell_3b_a[convoluted_grid]

    # Add the 8th one: output is in [0, 5]
    convoluted_grid = conv_with_hand_padding(grid, weights_method_3b_b, do_padded_fix)

    grid_b = grid_a + convoluted_grid

    # Apply a TLU: input in [0, 5], output in [0, 4]
    grid_c = table_cp_next_cell_3b_b[grid_b]

    # Add center
    grid = grid_c + grid

    # And a last TLU: input in [0, 5] and output in [0, 1]
    grid = table_cp_next_cell_3b_c[grid]

    return grid


@fhe.compiler({"grid": "encrypted"})
def update_grid_method_4b(grid):
    # Method which uses a first TLU of 4 bits and a second TLU of 2 bits

    weights_method_4b = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]
    )
    table_next_cell_4b_a = [i - 1 if i in [2, 3] else 0 for i in range(9)]
    table_next_cell_4b_b = [int(i in [2, 3]) for i in range(4)]

    table_cp_next_cell_4b_a = fhe.LookupTable(table_next_cell_4b_a)
    table_cp_next_cell_4b_b = fhe.LookupTable(table_next_cell_4b_b)

    # This is to workaround the fact that we have no pad option in fhe.conv
    do_padded_fix = True

    convoluted_grid = conv_with_hand_padding(grid, weights_method_4b, do_padded_fix)

    grid_a = table_cp_next_cell_4b_a[convoluted_grid]
    grid = grid_a + grid
    grid = table_cp_next_cell_4b_b[grid]

    return grid


@fhe.compiler({"grid": "encrypted"})
def update_grid_method_5b(grid):
    # Method which uses a single TLU of 5 bits

    weights_method_5b = np.array(
        [
            [1, 1, 1],
            [1, 9, 1],
            [1, 1, 1],
        ]
    )
    table_next_cell_5b = [int(i in [3, 9 + 2, 9 + 3]) for i in range(18)]

    table_cp_next_cell_5b = fhe.LookupTable(table_next_cell_5b)

    # This is to workaround the fact that we have no pad option in fhe.conv
    do_padded_fix = True

    convoluted_grid = conv_with_hand_padding(grid, weights_method_5b, do_padded_fix)

    grid = table_cp_next_cell_5b[convoluted_grid]

    return grid


@fhe.compiler({"grid": "encrypted"})
def update_grid_method_bits(grid):
    # Method which uses bits operator, with 4 calls to fhe.bits
    debug = False

    weights_method_basic = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]
    )

    # This is to workaround the fact that we have no pad option in fhe.conv
    do_padded_fix = True

    grid_dup = fhe.univariate(lambda x: x)(grid)
    convoluted_grid = conv_with_hand_padding(grid_dup, weights_method_basic, do_padded_fix)

    # Method with bits: 3 calls
    n = convoluted_grid

    s = 10 - n
    bs = 1 - fhe.bits(s)[3]

    t = 9 - n
    bt = fhe.bits(t)[3]

    u = 11 - n
    bu = 1 - fhe.bits(u)[3]

    # This is what it computes
    if debug:
        assert np.array_equal(bs, (n > 2).astype(np.int8)), f"{n=} {s=}"
        assert np.array_equal(bt, (n < 2).astype(np.int8)), f"{n=} {t=}"
        assert np.array_equal(bu, (n > 3).astype(np.int8)), f"{n=} {u=}"

    # Extract information
    n_is_2 = 1 - bs - bt
    n_is_2_or_3 = 1 - bu - bt

    # This is what it computes
    if debug:
        assert np.array_equal(n_is_2, (n == 2).astype(np.int8))
        assert np.array_equal(n_is_2_or_3, ((n == 2) | (n == 3)).astype(np.int8))

    # Update the grid
    new_grid = fhe.bits(2 * n_is_2_or_3 - n_is_2 + grid_dup)[1]

    return new_grid


@fhe.compiler({"grid": "encrypted"})
def update_grid_basic(grid):
    # Method which follows the naive approach

    weights_method_basic = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]
    )
    table_next_cell_basic_a = [int(i in [3]) for i in range(9)]
    table_next_cell_basic_b = [int(i in [2, 3]) for i in range(9)]

    table_cp_next_cell_basic_a = fhe.LookupTable(table_next_cell_basic_a)
    table_cp_next_cell_basic_b = fhe.LookupTable(table_next_cell_basic_b)

    # This is to workaround the fact that we have no pad option in fhe.conv
    do_padded_fix = True

    convoluted_grid = conv_with_hand_padding(grid, weights_method_basic, do_padded_fix)

    grid = table_cp_next_cell_basic_a[convoluted_grid] | (
        table_cp_next_cell_basic_b[convoluted_grid] & (grid == 1)
    )

    return grid


# Function for Game of Life
def update_grid(grid, method="method_3b"):
    assert grid.ndim == 4

    if method == "method_basic":
        return update_grid_basic(grid)

    if method == "method_3b":
        return update_grid_method_3b(grid)

    if method == "method_4b":
        return update_grid_method_4b(grid)

    if method == "method_5b":
        return update_grid_method_5b(grid)

    if method == "method_bits":
        return update_grid_method_bits(grid)

    msg = "Bad method"
    raise ValueError(msg)


# Graphic functions
# The graphical functions of this code were inspired by those of
# https://github.com/matheusgomes28/pygame-life/blob/main/pygame_life.py
# pylint: disable=unused-argument
def manage_graphics_and_refresh(
    grid,
    count,
    dimension,
    nb_initial_points,
    border_size,
    screen,
    background_refresh_color,
    background_color,
    life_color,
    time_new_grid_sleep,
    time_sleep,
    refresh_every,
    do_text_output,
):
    make_new_grid = count == 0 or (refresh_every > 0 and (count % refresh_every) == 0)

    count += 1

    # Refresh the grid from time to time
    if make_new_grid:
        grid = np.random.randint(2, size=(1, 1, dimension, dimension), dtype=np.int8)
        screen.fill(background_refresh_color)
        pygame.display.flip()
        time.sleep(time_new_grid_sleep)

    screen.fill(background_color)

    # Draw the grid
    width = grid.shape[2 + 0]
    height = grid.shape[2 + 1]
    cell_width = screen.get_width() / width
    cell_height = screen.get_height() / height

    for x in range(width):
        for y in range(height):
            if grid[0, 0, x, y]:
                pygame.draw.rect(
                    screen,
                    life_color,
                    (
                        x * cell_width + border_size,
                        y * cell_height + border_size,
                        cell_width - border_size,
                        cell_height - border_size,
                    ),
                )

    if do_text_output:
        np.set_printoptions(threshold=sys.maxsize, linewidth=np.nan)
        print(
            str(grid[0, 0, :, :])
            .replace("[", " ")
            .replace("]", " ")
            .replace("0", ".")
            .replace("1", "*")
            .replace(" ", "")
        )

    pygame.display.flip()

    # Make a pause for controlled speed
    time.sleep(time_sleep)

    return grid, count


def autotest(dimension):
    # Check all our methods return the same result

    for _ in range(100):
        # Take a random grid
        grid = np.random.randint(2, size=(1, 1, dimension, dimension), dtype=np.int8)

        # Check the results are the same
        results = {}

        for method in ["method_3b", "method_4b", "method_5b", "method_bits", "method_basic"]:
            results[method] = update_grid(grid, method=method)

        keys = list(results.keys())

        for k in keys[1:]:
            diff = results[keys[0]] - results[k]
            assert np.array_equal(
                results[keys[0]], results[k]
            ), f"\n{results[keys[0]]} \n{results[k]} are different, diff is \n{diff}"

    print("Tests of methods looks ok")


def manage_args():
    parser = argparse.ArgumentParser(description="Game of Life in Concrete Python.")
    parser.add_argument(
        "--dimension",
        dest="dimension",
        action="store",
        type=int,
        default=100,
        help="Dimension of the grid",
    )
    parser.add_argument(
        "--refresh_every",
        dest="refresh_every",
        action="store",
        type=int,
        default=None,
        help="Refresh the grid every X steps",
    )
    parser.add_argument(
        "--method",
        dest="method",
        action="store",
        choices=["method_3b", "method_4b", "method_5b", "method_bits", "method_basic"],
        default="method_5b",
        help="Method for refreshing the grid",
    )
    parser.add_argument(
        "--log2_global_p_error",
        dest="log2_global_p_error",
        action="store",
        type=float,
        default=None,
        help="Probability of correctness issue (full circuit)",
    )
    parser.add_argument(
        "--log2_p_error",
        dest="log2_p_error",
        action="store",
        type=float,
        default=-16,
        help="Probability of correctness issue (individual TLU)",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        dest="fhe_simulation",
        help="Simulate instead of running computations in FHE",
    )
    parser.add_argument(
        "--show_mlir",
        action="store_true",
        dest="show_mlir",
        help="Show the MLIR",
    )
    parser.add_argument(
        "--show_graph",
        action="store_true",
        dest="show_graph",
        help="Show the graph",
    )
    parser.add_argument(
        "--verbose_compilation",
        action="store_true",
        dest="verbose_compilation",
        help="Add verbose option in compilation",
    )
    parser.add_argument(
        "--stop_after_compilation",
        action="store_true",
        dest="stop_after_compilation",
        help="Stop after compilation",
    )
    parser.add_argument(
        "--text_output",
        action="store_true",
        dest="text_output",
        help="Print a text output of the grid",
    )
    parser.add_argument(
        "--seed",
        action="store",
        dest="seed",
        type=int,
        default=np.random.randint(2**32 - 1),
        help="Set a seed",
    )

    args = parser.parse_args()
    print(f"Using seed {args.seed=}")

    np.random.seed(args.seed)
    return args


def main():
    # Options by the user
    args = manage_args()

    # Dimension of the grid. The larger, the slower, in FHE
    dimension = args.dimension

    # Which method
    which_method = args.method

    # Switch this off to not compile in FHE
    do_compile = True

    # Activate to simulate
    fhe_simulation = args.fhe_simulation

    # Refresh with a random grid every X steps
    refresh_every = min(100, dimension) if args.refresh_every is None else args.refresh_every

    # To see the execution time
    do_print_time = True

    # If there is no X server
    do_text_output = args.text_output

    # Probability of failure
    log2_global_p_error = args.log2_global_p_error
    log2_p_error = args.log2_p_error

    # Options for graphics
    nb_initial_points = dimension**2
    size = (1000, 700)
    background_color = (20, 20, 20)
    background_refresh_color = (150, 20, 20)
    life_color = (55, 200, 200)
    border_size = 1

    time_sleep = 0.1 if not do_compile or fhe_simulation else 0

    time_new_grid_sleep = 0.4

    # Autotest
    autotest(dimension=dimension)

    print(f"Using method {which_method}")
    print(f"Using a grid {dimension} * {dimension}")
    print(f"Refreshing every {refresh_every} steps")
    print(f"Using 2**{log2_global_p_error} for global_p_error")
    print(f"Using 2**{log2_p_error} for p_error")

    # Compile
    if do_compile:
        inputset = [
            np.random.randint(2, size=(1, 1, dimension, dimension), dtype=np.int8)
            for _ in range(1000)
        ]

        if which_method == "method_3b":
            function = update_grid_method_3b
        elif which_method == "method_4b":
            function = update_grid_method_4b
        elif which_method == "method_5b":
            function = update_grid_method_5b
        elif which_method == "method_bits":
            function = update_grid_method_bits
        else:
            assert which_method == "method_basic"
            function = update_grid_basic

        circuit = function.compile(
            inputset,
            show_mlir=args.show_mlir,
            show_graph=args.show_graph,
            fhe_simulation=fhe_simulation,
            global_p_error=None,  # 2**log2_global_p_error,
            p_error=2**log2_p_error,
            bitwise_strategy_preference=fhe.BitwiseStrategy.ONE_TLU_PROMOTED,
            verbose=args.verbose_compilation,
            # parameter_selection_strategy=fhe.ParameterSelectionStrategy.MULTI,
            # single_precision=False,
        )

        # print(circuit.graph.format(show_assigned_bit_widths=True))

        if args.stop_after_compilation:
            sys.exit(0)

    # Set plot
    pygame.init()
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Game of Life in Concrete Python")
    count = 0
    grid = None

    # Run the key generation, to avoid to have a first execution time which is slower
    if do_compile and not fhe_simulation:
        time_start = time.time()
        circuit.keygen()
        time_end = time.time()

        if do_print_time:
            print(f"Generating key in {time_end - time_start:.2f} seconds")

    while True:
        if pygame.QUIT in [e.type for e in pygame.event.get()]:
            sys.exit(0)

        grid, count = manage_graphics_and_refresh(
            grid,
            count,
            dimension,
            nb_initial_points,
            border_size,
            screen,
            background_refresh_color,
            background_color,
            life_color,
            time_new_grid_sleep,
            time_sleep,
            refresh_every,
            do_text_output,
        )

        # Update the grid
        time_start = time.time()

        if do_compile:
            grid = circuit.simulate(grid) if fhe_simulation else circuit.encrypt_run_decrypt(grid)
        else:
            grid = update_grid(grid, method=which_method)

        time_end = time.time()

        if do_print_time:
            print(f"Updating grid in {time_end - time_start:.2f} seconds")

        assert np.min(grid) >= 0
        assert np.max(grid) <= 1


if __name__ == "__main__":
    main()
