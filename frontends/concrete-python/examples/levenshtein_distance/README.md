# Computing the Levenshtein distance in FHE

## Levenshtein distance

Levenshtein distance is a classical distance to compare two strings. Let's write strings a and b as
vectors of characters, meaning a[0] is the first char of a and a[1:] is the rest of the string.
Levenshtein distance is defined as:

    Levenshtein(a, b) :=
        length(a) if length(b) == 0, or
        length(b) if length(a) == 0, or
        Levenshtein(a[1:], b[1:]) if a[0] == b[0], or
        1 + min(Levenshtein(a[1:], b), Levenshtein(a, b[1:]), Levenshtein(a[1:], b[1:]))

More information can be found for example on the [Wikipedia page](https://en.wikipedia.org/wiki/Levenshtein_distance).

## Computing the distance in FHE

It can be interesting to compute this distance over encrypted data, for example in the banking sector.
We show in [our code](levenshtein_distance.py) how to do that simply, with our FHE modules.

Available options are:

```
usage: levenshtein_distance.py [-h] [--show_mlir] [--show_optimizer] [--autotest] [--autoperf] [--distance DISTANCE DISTANCE]
                               [--alphabet {string,STRING,StRiNg,ACTG}] [--max_string_length MAX_STRING_LENGTH]

Levenshtein distance in Concrete.

optional arguments:
  -h, --help            show this help message and exit
  --show_mlir           Show the MLIR
  --show_optimizer      Show the optimizer outputs
  --autotest            Run random tests
  --autoperf            Run benchmarks
  --distance DISTANCE DISTANCE
                        Compute a distance
  --alphabet {string,STRING,StRiNg,ACTG}
                        Setting the alphabet
  --max_string_length MAX_STRING_LENGTH
                        Setting the maximal size of strings
```

The different alphabets are:
- string: non capitalized letters, ie `[a-z]*`
- STRING: capitalized letters, ie `[A-Z]*`
- StRiNg: non capitalized letters and capitalized letters
- ACTG: `[ACTG]*`, for DNA analysis

It is very easy to add a new alphabet in the code.

The most important usages are:

- `python levenshtein_distance.py --distance Zama amazing --alphabet StRiNg --max_string_length 7`: Compute the distance between
strings "Zama" and "amazing", considering the chars of "StRiNg" alphabet

```

Running distance between strings 'Zama' and 'amazing' for alphabet StRiNg:

    Computing Levenshtein between strings 'Zama' and 'amazing' - distance is 5, computed in 44.51 seconds

Successful end
```

- `python levenshtein_distance.py --autotest`: Run random tests with the alphabet.

```
Making random tests with alphabet string
Letters are abcdefghijklmnopqrstuvwxyz

Computations in simulation

    Computing Levenshtein between strings '' and '' - OK
    Computing Levenshtein between strings '' and 'p' - OK
    Computing Levenshtein between strings '' and 'vv' - OK
    Computing Levenshtein between strings '' and 'mxg' - OK
    Computing Levenshtein between strings '' and 'iuxf' - OK
    Computing Levenshtein between strings 'k' and '' - OK
    Computing Levenshtein between strings 'p' and 'g' - OK
    Computing Levenshtein between strings 'v' and 'ky' - OK
    Computing Levenshtein between strings 'f' and 'uoq' - OK
    Computing Levenshtein between strings 'f' and 'kwfj' - OK
    Computing Levenshtein between strings 'ut' and '' - OK
    Computing Levenshtein between strings 'pa' and 'g' - OK
    Computing Levenshtein between strings 'bu' and 'sx' - OK
    Computing Levenshtein between strings 'is' and 'diy' - OK
    Computing Levenshtein between strings 'fz' and 'unda' - OK
    Computing Levenshtein between strings 'sem' and '' - OK
    Computing Levenshtein between strings 'dbr' and 'o' - OK
    Computing Levenshtein between strings 'dgj' and 'hk' - OK
    Computing Levenshtein between strings 'ejb' and 'tfo' - OK
    Computing Levenshtein between strings 'afa' and 'ygqo' - OK
    Computing Levenshtein between strings 'lhcc' and '' - OK
    Computing Levenshtein between strings 'uoiu' and 'u' - OK
    Computing Levenshtein between strings 'tztt' and 'xo' - OK
    Computing Levenshtein between strings 'ufsa' and 'mil' - OK
    Computing Levenshtein between strings 'uuzl' and 'dzkr' - OK

Computations in FHE

    Computing Levenshtein between strings '' and '' - OK in 1.29 seconds
    Computing Levenshtein between strings '' and 'p' - OK in 0.26 seconds
    Computing Levenshtein between strings '' and 'vv' - OK in 0.26 seconds
    Computing Levenshtein between strings '' and 'mxg' - OK in 0.22 seconds
    Computing Levenshtein between strings '' and 'iuxf' - OK in 0.22 seconds
    Computing Levenshtein between strings 'k' and '' - OK in 0.22 seconds
    Computing Levenshtein between strings 'p' and 'g' - OK in 1.09 seconds
    Computing Levenshtein between strings 'v' and 'ky' - OK in 1.93 seconds
    Computing Levenshtein between strings 'f' and 'uoq' - OK in 3.09 seconds
    Computing Levenshtein between strings 'f' and 'kwfj' - OK in 3.98 seconds
    Computing Levenshtein between strings 'ut' and '' - OK in 0.25 seconds
    Computing Levenshtein between strings 'pa' and 'g' - OK in 1.90 seconds
    Computing Levenshtein between strings 'bu' and 'sx' - OK in 3.52 seconds
    Computing Levenshtein between strings 'is' and 'diy' - OK in 5.04 seconds
    Computing Levenshtein between strings 'fz' and 'unda' - OK in 6.53 seconds
    Computing Levenshtein between strings 'sem' and '' - OK in 0.22 seconds
    Computing Levenshtein between strings 'dbr' and 'o' - OK in 2.78 seconds
    Computing Levenshtein between strings 'dgj' and 'hk' - OK in 4.92 seconds
    Computing Levenshtein between strings 'ejb' and 'tfo' - OK in 7.18 seconds
    Computing Levenshtein between strings 'afa' and 'ygqo' - OK in 9.25 seconds
    Computing Levenshtein between strings 'lhcc' and '' - OK in 0.22 seconds
    Computing Levenshtein between strings 'uoiu' and 'u' - OK in 3.52 seconds
    Computing Levenshtein between strings 'tztt' and 'xo' - OK in 6.45 seconds
    Computing Levenshtein between strings 'ufsa' and 'mil' - OK in 9.11 seconds
    Computing Levenshtein between strings 'uuzl' and 'dzkr' - OK in 12.01 seconds

Successful end
```

- `python levenshtein_distance.py --autoperf`: Benchmark with random strings, for the different alphabets.

```
Typical performances for alphabet ACTG, with string of maximal length:

    Computing Levenshtein between strings 'GCGA' and 'GTCA' - OK in 6.04 seconds
    Computing Levenshtein between strings 'TCGA' and 'ACAA' - OK in 5.57 seconds
    Computing Levenshtein between strings 'CAGT' and 'CGTT' - OK in 5.63 seconds

Typical performances for alphabet string, with string of maximal length:

    Computing Levenshtein between strings 'ctow' and 'qtor' - OK in 17.54 seconds
    Computing Levenshtein between strings 'vwky' and 'enfh' - OK in 16.46 seconds
    Computing Levenshtein between strings 'dqse' and 'spps' - OK in 16.49 seconds

Typical performances for alphabet STRING, with string of maximal length:

    Computing Levenshtein between strings 'TQBW' and 'LKIZ' - OK in 16.62 seconds
    Computing Levenshtein between strings 'HANA' and 'CFVO' - OK in 16.32 seconds
    Computing Levenshtein between strings 'BEXY' and 'YAWM' - OK in 16.58 seconds

Typical performances for alphabet StRiNg, with string of maximal length:

    Computing Levenshtein between strings 'iYmH' and 'ONnz' - OK in 30.56 seconds
    Computing Levenshtein between strings 'hZyX' and 'vhHH' - OK in 30.11 seconds
    Computing Levenshtein between strings 'sJdj' and 'strn' - OK in 30.48 seconds

Successful end
```

## Complexity analysis

Let's analyze a bit the complexity of the function `levenshtein_fhe` in FHE. We can see that the
function cannot apply `if`'s as in the clear function `levenshtein_clear`: it has to compute the two
branches (the one for the True, and the one for the False), and finally compute an `fhe.if_then_else`
of the two possible values. This slowdown is not specific to Concrete, it is by nature of FHE, where
encrypted conditions imply such a trick.

Another interesting part is the impact of the choice of the alphabet: in `run`, we are going to
compare two chars of the alphabet, and return an encrypted boolean to code for the equality / inequality
of these two chars. This is basically done with a single programmable bootstrapping (PBS) of `w+1`
bits, where `w` is the floored log2 value of the number of chars in the alphabet. For example, for
the 'string' alphabet, which has 26 letters, `w = 5` and so we use a signed 6-bit value as input of a
table lookup. For the larger 'StRiNg' alphabet, that's a signed 7-bit PBS. For small DNA alphabet 'ACTG',
it's only signed 3-bit PBS.

## Benchmarks on hpc7a

The benchmarks were done using Concrete 2.7 on `hpc7a` machine on AWS, and give:

```
Typical performances for alphabet ACTG, with string of maximal length:

    Computing Levenshtein between strings 'AGTC' and 'TGGA' - OK in 6.00 seconds
    Computing Levenshtein between strings 'GTAA' and 'AGAC' - OK in 5.51 seconds
    Computing Levenshtein between strings 'TCTT' and 'CACG' - OK in 5.49 seconds

Typical performances for alphabet string, with string of maximal length:

    Computing Levenshtein between strings 'jqdk' and 'zqlf' - OK in 17.43 seconds
    Computing Levenshtein between strings 'uquc' and 'qvvp' - OK in 16.50 seconds
    Computing Levenshtein between strings 'vebm' and 'ybqo' - OK in 16.46 seconds

Typical performances for alphabet STRING, with string of maximal length:

    Computing Levenshtein between strings 'UQES' and 'NWXQ' - OK in 16.53 seconds
    Computing Levenshtein between strings 'LAJG' and 'NEGP' - OK in 16.26 seconds
    Computing Levenshtein between strings 'OSQG' and 'OTEH' - OK in 16.52 seconds

Typical performances for alphabet StRiNg, with string of maximal length:

    Computing Levenshtein between strings 'ixgu' and 'cOSy' - OK in 30.94 seconds
    Computing Levenshtein between strings 'QGCj' and 'Lknx' - OK in 29.82 seconds
    Computing Levenshtein between strings 'fKVC' and 'xqaI' - OK in 30.27 seconds

Successful end
```
