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

We can run our `prime-match.py` to perform the computations: `FHE Simulation` is done in the clear to build expected results, while `FHE` is the real FHE computation. Our execution here was done on an `hpc7a` machine on AWS, with Concrete FIXME.

```
$ python prime-match.py


FIXME: run that on hpc7a machine
```

## Executing the semi honest protocol

We have executed the semi-honest protocol, still on an `hpc7a` machine on AWS, with Concrete FIXME.


```
$ python prime-match-semi-honest.py

FIXME: run that on hpc7a machine
```
