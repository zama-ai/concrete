import random
import time
from enum import IntEnum, auto
from typing import Set

import numpy as np

from concrete import fhe

NUMBER_OF_BANK_ORDERS = 10
NUMBER_OF_CLIENT_ORDERS = 5

MINIMUM_ORDER_QUANTITY = 5
MAXIMUM_ORDER_QUANTITY = 60

assert 0 < MINIMUM_ORDER_QUANTITY < MAXIMUM_ORDER_QUANTITY


class OrderType(IntEnum):
    Buy = 0
    Sell = auto()


class OrderSymbol(IntEnum):
    A = 0
    B = auto()
    C = auto()
    D = auto()
    E = auto()
    F = auto()
    G = auto()
    H = auto()
    I = auto()
    J = auto()
    # K = auto()
    # L = auto()
    # M = auto()
    # N = auto()
    # O = auto()
    # P = auto()
    # Q = auto()
    # R = auto()
    # S = auto()
    # T = auto()
    # U = auto()
    # V = auto()
    # W = auto()
    # X = auto()
    # Y = auto()
    # Z = auto()


class Order:
    def __init__(self, order_type: OrderType, order_symbol: OrderSymbol, order_quantity: int):
        self.order_type = order_type
        self.order_symbol = order_symbol
        self.order_quantity = order_quantity

    def __str__(self):
        return f"{self.order_type.name:>4} {self.order_quantity:3} of {self.order_symbol.name}"

    def __repr__(self):
        return f"{repr(self.order_type)} {self.order_quantity} of {repr(self.order_symbol)}"

    @staticmethod
    def random(skiplist: Set[OrderSymbol]) -> "Order":
        order_type = random.choice(list(OrderType))
        order_symbol = random.choice(list(set(OrderSymbol) - skiplist))
        order_quantity = random.randint(MINIMUM_ORDER_QUANTITY, MAXIMUM_ORDER_QUANTITY)
        return Order(order_type, order_symbol, order_quantity)


def prime_match(
    bank_order_types,
    bank_order_symbols,
    bank_order_quantities,
    client_order_types,
    client_order_symbols,
    client_order_quantities,
):
    with fhe.tag("comparing-order-types"):
        types_differ = bank_order_types.reshape(-1, 1) != client_order_types

    with fhe.tag("comparing-order-symbols"):
        symbols_match = bank_order_symbols.reshape(-1, 1) == client_order_symbols

    with fhe.tag("checking-if-order-can-be-filled"):
        can_fill = symbols_match & types_differ

    with fhe.tag("calculating-matching-order-quantity"):
        matching_quantity = np.minimum(
            bank_order_quantities.reshape(-1, 1), client_order_quantities
        )

    with fhe.tag("calculating-filled-order-quantity"):
        filled_quantity = fhe.multivariate(lambda x, y: x * y)(can_fill, matching_quantity)

    with fhe.tag("creating-result"):
        result = fhe.zeros((NUMBER_OF_BANK_ORDERS + NUMBER_OF_CLIENT_ORDERS, 1 + 1 + 1))

        result[0:NUMBER_OF_BANK_ORDERS, 0] = bank_order_types
        result[0:NUMBER_OF_BANK_ORDERS, 1] = bank_order_symbols

        result[NUMBER_OF_BANK_ORDERS:, 0] = client_order_types
        result[NUMBER_OF_BANK_ORDERS:, 1] = client_order_symbols

        result[:NUMBER_OF_BANK_ORDERS, 2] = np.sum(filled_quantity, axis=1)
        result[NUMBER_OF_BANK_ORDERS:, 2] = np.sum(filled_quantity, axis=0)

        return result


inputset = [
    (
        np.random.randint(0, len(OrderType), size=(NUMBER_OF_BANK_ORDERS,)),
        np.array(
            [int(symbol) for symbol in random.sample(list(OrderSymbol), NUMBER_OF_BANK_ORDERS)]
        ),
        np.random.randint(
            MINIMUM_ORDER_QUANTITY,
            MAXIMUM_ORDER_QUANTITY + 1,
            size=(NUMBER_OF_BANK_ORDERS,),
        ),
        np.random.randint(0, len(OrderType), size=(NUMBER_OF_CLIENT_ORDERS,)),
        np.array(
            [int(symbol) for symbol in random.sample(list(OrderSymbol), NUMBER_OF_CLIENT_ORDERS)]
        ),
        np.random.randint(
            MINIMUM_ORDER_QUANTITY,
            MAXIMUM_ORDER_QUANTITY + 1,
            size=(NUMBER_OF_CLIENT_ORDERS,),
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
        "bank_order_types": "encrypted",
        "bank_order_symbols": "encrypted",
        "bank_order_quantities": "encrypted",
        "client_order_types": "encrypted",
        "client_order_symbols": "encrypted",
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


sample_bank_orders = []

blacklist: Set[OrderSymbol] = set()
for _ in range(NUMBER_OF_BANK_ORDERS):
    order = Order.random(blacklist)
    blacklist.add(order.order_symbol)
    sample_bank_orders.append(order)

sample_client_orders = []

blacklist: Set[OrderSymbol] = set()
for _ in range(NUMBER_OF_CLIENT_ORDERS):
    order = Order.random(blacklist)
    blacklist.add(order.order_symbol)
    sample_client_orders.append(order)

print("Sample Input:")
print()

print("\tBank Orders:")
for order in sample_bank_orders:
    print(f"\t\t{order}")
print()

print("\tClient Orders:")
for order in sample_client_orders:
    print(f"\t\t{order}")
print()

print()

sample_bank_order_types = [int(order.order_type) for order in sample_bank_orders]
sample_bank_order_symbols = [int(order.order_symbol) for order in sample_bank_orders]
sample_bank_order_quantities = [order.order_quantity for order in sample_bank_orders]

sample_client_order_types = [int(order.order_type) for order in sample_client_orders]
sample_client_order_symbols = [int(order.order_symbol) for order in sample_client_orders]
sample_client_order_quantities = [order.order_quantity for order in sample_client_orders]

sample_args = [
    sample_bank_order_types,
    sample_bank_order_symbols,
    sample_bank_order_quantities,
    sample_client_order_types,
    sample_client_order_symbols,
    sample_client_order_quantities,
]


def construct_result(matches):
    raw_bank_orders = matches[0:NUMBER_OF_BANK_ORDERS]
    raw_client_orders = matches[NUMBER_OF_BANK_ORDERS:]

    bank_orders = [
        Order(OrderType(raw_order_type), OrderSymbol(raw_order_symbol), order_quantity)
        for raw_order_type, raw_order_symbol, order_quantity in raw_bank_orders
    ]
    client_orders = [
        Order(OrderType(raw_order_type), OrderSymbol(raw_order_symbol), order_quantity)
        for raw_order_type, raw_order_symbol, order_quantity in raw_client_orders
    ]

    return bank_orders, client_orders


print("Simulated Output:")
simulated_matches = circuit.simulate(*sample_args)
simulated_bank_result, simulated_client_result = construct_result(simulated_matches)
print()

print("\tBank Orders:")
for order in simulated_bank_result:
    print(f"\t\t{order}")
print()

print("\tClient Orders:")
for order in simulated_client_result:
    print(f"\t\t{order}")
print()

print()

print("Actual Output:")
start = time.time()
executed_matches = circuit.encrypt_run_decrypt(*sample_args)
end = time.time()
executed_bank_result, executed_client_result = construct_result(executed_matches)
print()

print("\tBank Orders:")
for order in executed_bank_result:
    print(f"\t\t{order}")
print()

print("\tClient Orders:")
for order in executed_client_result:
    print(f"\t\t{order}")
print()

print(f"Complexity was: {circuit.complexity:.3f}")
print()
print(f"Nb of transactions: {NUMBER_OF_BANK_ORDERS*NUMBER_OF_CLIENT_ORDERS}")
print(f"Nb of Symbols: {len(OrderSymbol)}")
print(f"Execution took: {end - start:.3f} seconds")
