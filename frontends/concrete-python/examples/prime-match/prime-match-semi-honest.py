import random
import time

import numpy as np

from concrete import fhe

# ruff: noqa:S311

# Users can change these settings
NUMBER_OF_SYMBOLS = 10
MINIMUM_ORDER_QUANTITY = 1
MAXIMUM_ORDER_QUANTITY = 50

assert 0 < MINIMUM_ORDER_QUANTITY < MAXIMUM_ORDER_QUANTITY


def prime_match(
    bank_order_quantities,
    client_order_quantities,
):
    with fhe.tag("calculating-matching-order-quantity"):
        return np.minimum(bank_order_quantities, client_order_quantities)


inputset = [
    (
        np.random.randint(
            MINIMUM_ORDER_QUANTITY, MAXIMUM_ORDER_QUANTITY, size=(NUMBER_OF_SYMBOLS * 2,)
        ),
        np.random.randint(
            MINIMUM_ORDER_QUANTITY, MAXIMUM_ORDER_QUANTITY, size=(NUMBER_OF_SYMBOLS * 2,)
        ),
    )
    for _ in range(1000)
]
configuration = fhe.Configuration(
    enable_unsafe_features=True,
    use_insecure_key_cache=True,
    insecure_key_cache_location=".keys",
    fhe_simulation=True,
    min_max_strategy_preference=fhe.MinMaxStrategy.ONE_TLU_PROMOTED,
    show_progress=True,
    progress_tag=True,
)


# Only the quantities are encrypted, on both client and bank sides
compiler = fhe.Compiler(
    prime_match,
    {
        "bank_order_quantities": "encrypted",
        "client_order_quantities": "encrypted",
    },
)
circuit = compiler.compile(inputset, configuration)


print()
start = time.time()
assert circuit.keys is not None
circuit.keys.generate()
end = time.time()
print(f"Key generation took: {end - start:.3f} seconds")
print()


# Generate random orders
sample_bank_order_quantities = [0] * NUMBER_OF_SYMBOLS * 2
sample_client_order_quantities = [0] * NUMBER_OF_SYMBOLS * 2

for i in range(NUMBER_OF_SYMBOLS):
    # Randomly choose between a Sell Client / Buy Bank or a Buy Client / Sell Bank
    # but avoir Sell Client / Buy Client
    idx = i + random.randint(0, 1) * NUMBER_OF_SYMBOLS
    sample_bank_order_quantities[idx] = np.random.randint(
        MINIMUM_ORDER_QUANTITY, MAXIMUM_ORDER_QUANTITY
    )
    sample_client_order_quantities[idx] = np.random.randint(
        MINIMUM_ORDER_QUANTITY, MAXIMUM_ORDER_QUANTITY
    )

sample_args = (sample_bank_order_quantities, sample_client_order_quantities)

# Perform the matching with FHE simulation
print("FHE Simulation:")
simulated_matches = circuit.simulate(*sample_args)
print()

print("\tResult Orders:")
for c1_order, c2_order, result in zip(
    sample_bank_order_quantities, sample_client_order_quantities, simulated_matches
):
    print(f"\t\t{c1_order}\t{c2_order}\t->\t{result}")
print()

# Perform the matching in FHE
print("FHE:")
start = time.time()
executed_matches = circuit.encrypt_run_decrypt(*sample_args)
end = time.time()
print()

print("\tResult Orders:")
for c1_order, c2_order, result in zip(
    sample_bank_order_quantities, sample_client_order_quantities, executed_matches
):
    print(f"\t\t{c1_order}\t{c2_order}\t->\t{result}")
print()

# Check
assert all(simulated_matches == executed_matches), "Error in FHE computation"

# Some information about the complexity of the computations
NUMBER_OF_TRANSACTIONS = NUMBER_OF_SYMBOLS * 2
print(f"Complexity was: {circuit.complexity:.3f}")
print()
print(f"Quantities in [{MINIMUM_ORDER_QUANTITY}, {MAXIMUM_ORDER_QUANTITY}]")
print(f"Nb of transactions: {NUMBER_OF_TRANSACTIONS}")
print(f"Nb of Symbols: {NUMBER_OF_SYMBOLS}")
print(
    f"Execution took: {end - start:.3f} seconds, ie "
    f"{(end - start) / NUMBER_OF_TRANSACTIONS:.3f} seconds per transaction"
)
