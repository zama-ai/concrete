# Inventory Matching System

## Introduction

This tutorial implements two variants of the Inventory Matching System described in this [paper](https://eprint.iacr.org/2023/400.pdf):
`prime-match.py` implements the classical protocol, while `prime-match-semi-honest.py` follows the semi-honest protocol.

The principle is as follows: a bank has a list of orders, and a client has another list of orders. Orders are either `Sell` or `Buy`, followed by an asset. We want to apply the matching between the orders, without the parties to know what are each other orders.

A simple example is

```
	Bank Orders:
		Sell  10 of C
		 Buy  47 of A
		Sell  31 of D

	Client Orders:
		Sell  50 of A
		Sell  24 of B
		 Buy  18 of D
```

The corresponding resolution is

```
	Bank Orders:
		Sell   0 of C
		 Buy  47 of A
		Sell  18 of D

	Client Orders:
		Sell  47 of A
		Sell  24 of B
		 Buy  18 of D
```

## Executing the classic protocol

We can run our `prime-match.py` to perform the computations: `FHE Simulation` is done in the clear to build expected results, while `FHE` is the real FHE computation. Our execution here was done on an `hpc7a` machine on AWS, with Concrete v2.8.1, in about 3 seconds for 50 transactions on 10 symbols.

```
$ python prime-match.py

Key generation took: 10.746 seconds

Sample Input:

	Bank Orders:
		 Buy  17 of E
		 Buy  30 of F
		Sell   6 of D
		Sell  34 of J
		 Buy  30 of C
		Sell  54 of I
		 Buy  25 of G
		Sell   8 of B
		Sell  21 of H
		 Buy  24 of A

	Client Orders:
		 Buy  56 of F
		 Buy  32 of A
		Sell  44 of H
		 Buy  39 of J
		Sell  50 of C


FHE Simulation:

	Bank Orders:
		 Buy   0 of E
		 Buy   0 of F
		Sell   0 of D
		Sell  34 of J
		 Buy  30 of C
		Sell   0 of I
		 Buy   0 of G
		Sell   0 of B
		Sell   0 of H
		 Buy   0 of A

	Client Orders:
		 Buy   0 of F
		 Buy   0 of A
		Sell   0 of H
		 Buy  34 of J
		Sell  30 of C


FHE:

	Bank Orders:
		 Buy   0 of E
		 Buy   0 of F
		Sell   0 of D
		Sell  34 of J
		 Buy  30 of C
		Sell   0 of I
		 Buy   0 of G
		Sell   0 of B
		Sell   0 of H
		 Buy   0 of A

	Client Orders:
		 Buy   0 of F
		 Buy   0 of A
		Sell   0 of H
		 Buy  34 of J
		Sell  30 of C

Complexity was: 158500926100.000

Nb of transactions: 50
Nb of Symbols: 10
Execution took: 3.237 seconds, ie 0.065 seconds per transaction
```

## Executing the semi honest protocol

We have executed the semi-honest protocol, still on an `hpc7a` machine on AWS, with Concrete v2.8.1, in 1 second for 20 transactions on 10 symbols.

```
$ python prime-match-semi-honest.py


Key generation took: 9.606 seconds

FHE Simulation:

	Result Orders:
		0	0	->	0
		23	20	->	20
		0	0	->	0
		7	27	->	7
		0	0	->	0
		0	0	->	0
		0	0	->	0
		41	13	->	13
		23	24	->	23
		0	0	->	0
		11	4	->	4
		0	0	->	0
		45	3	->	3
		0	0	->	0
		4	49	->	4
		33	31	->	31
		24	15	->	15
		0	0	->	0
		0	0	->	0
		2	23	->	2

FHE:

	Result Orders:
		0	0	->	0
		23	20	->	20
		0	0	->	0
		7	27	->	7
		0	0	->	0
		0	0	->	0
		0	0	->	0
		41	13	->	13
		23	24	->	23
		0	0	->	0
		11	4	->	4
		0	0	->	0
		45	3	->	3
		0	0	->	0
		4	49	->	4
		33	31	->	31
		24	15	->	15
		0	0	->	0
		0	0	->	0
		2	23	->	2

Complexity was: 23066687400.000

Quantities in [1, 50]
Nb of transactions: 20
Nb of Symbols: 10
Execution took: 1.011 seconds, ie 0.051 seconds per transaction
```
