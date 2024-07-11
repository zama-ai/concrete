import random
import time

import numpy as np

from concrete import fhe

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
    comparison_strategy_preference=fhe.ComparisonStrategy.ONE_TLU_PROMOTED,
    bitwise_strategy_preference=fhe.BitwiseStrategy.ONE_TLU_PROMOTED,
    multivariate_strategy_preference=fhe.MultivariateStrategy.PROMOTED,
    min_max_strategy_preference=fhe.MinMaxStrategy.ONE_TLU_PROMOTED,
    show_progress=True,
    progress_tag=True,
)


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
circuit.keys.generate()
end = time.time()
print(f"Key generation took: {end - start:.3f} seconds")
print()


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

print("Simulated Output:")
simulated_matches = circuit.simulate(*sample_args)
print()


print("\tResult Orders:")
for c1_order, c2_order, result in zip(
    sample_bank_order_quantities[:15], sample_client_order_quantities[:15], simulated_matches[:15]
):
    print(f"\t\t{c1_order}\t{c2_order}\t->\t{result}")
print()

print("Actual Output:")
start = time.time()
executed_matches = circuit.encrypt_run_decrypt(*sample_args)
end = time.time()
print()

print("\tResult Orders:")
for c1_order, c2_order, result in zip(
    sample_bank_order_quantities[:15], sample_client_order_quantities[:15], executed_matches[:15]
):
    print(f"\t\t{c1_order}\t{c2_order}\t->\t{result}")
print()

print(f"Complexity was: {circuit.complexity:.3f}")
print()
print(f"Quantities in [{MINIMUM_ORDER_QUANTITY}, {MAXIMUM_ORDER_QUANTITY}]")
print(f"Nb of transactions: {NUMBER_OF_SYMBOLS*2}")
print(f"Nb of Symbols: {NUMBER_OF_SYMBOLS}")
print(f"Execution took: {end - start:.3f} seconds")
