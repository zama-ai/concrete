# Prime Match Poc with Concrete Python

## Intro

This Poc implements two variants of the Inventory Matching System described in this [paper](https://eprint.iacr.org/2023/400.pdf).
`prime-match.py` implements the classical protocol, while `prime-match-sh.py` follows the semi-honest protocol described in the paper.

## Installation

```python
pip install -r requirements.txt
```

## Output classic protocol

Results are described in the stdout, `simulated output` is done in the clear to build expected results, while `actual output` is the result
using FHE:

```
$ python prime-match.py

KeySetCache: miss, regenerating .keys/13120680643775490630
Key generation took: 17.843 seconds

Sample Input:

	Bank Orders:
		Sell  50 of J
		 Buy  25 of H
		 Buy  53 of A
		Sell  43 of D
		 Buy  60 of E
		 Buy  28 of G
		 Buy  60 of B
		 Buy  25 of I
		Sell  38 of C
		 Buy  21 of F

	Client Orders:
		Sell  47 of A
		Sell  24 of B
		 Buy  57 of E
		 Buy  18 of D
		 Buy  50 of H


Simulated Output:
100% |██████████████████████████████████████████████████| 100%

	Bank Orders:
		Sell   0 of J
		 Buy   0 of H
		 Buy  47 of A
		Sell  18 of D
		 Buy   0 of E
		 Buy   0 of G
		 Buy  24 of B
		 Buy   0 of I
		Sell   0 of C
		 Buy   0 of F

	Client Orders:
		Sell  47 of A
		Sell  24 of B
		 Buy   0 of E
		 Buy  18 of D
		 Buy   0 of H


Actual Output:
100% |██████████████████████████████████████████████████| 100%

	Bank Orders:
		Sell   0 of J
		 Buy   0 of H
		 Buy  47 of A
		Sell  18 of D
		 Buy   0 of E
		 Buy   0 of G
		 Buy  24 of B
		 Buy   0 of I
		Sell   0 of C
		 Buy   0 of F

	Client Orders:
		Sell  47 of A
		Sell  24 of B
		 Buy   0 of E
		 Buy  18 of D
		 Buy   0 of H

Complexity was: 205423680000.000

Nb of transactions: 50
Nb of Symbols: 10
Execution took: 12.493 seconds
```

## Output Semi honest protocol

```
$ python prime-match-sh.py
Key generation took: 2.026 seconds

Simulated Output:
100% |██████████████████████████████████████████████████| 100%

	Result Orders:
		0	0	->	0
		0	0	->	0
		0	0	->	0
		0	0	->	0
		0	0	->	0
		10	36	->	10
		0	0	->	0
		0	0	->	0
		0	0	->	0
		4	27	->	4
		19	22	->	19
		38	44	->	38
		6	5	->	5
		46	30	->	30
		46	7	->	7

Actual Output:
100% |██████████████████████████████████████████████████| 100%

	Result Orders:
		0	0	->	0
		0	0	->	0
		0	0	->	0
		0	0	->	0
		0	0	->	0
		10	36	->	10
		0	0	->	0
		0	0	->	0
		0	0	->	0
		4	27	->	4
		19	22	->	19
		38	44	->	38
		6	5	->	5
		46	30	->	30
		46	7	->	7

Complexity was: 22568614200.000

Quantities in [1, 50]
Nb of transactions: 20
Nb of Symbols: 10
Execution took: 1.646 seconds
```
