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

- `levenshtein_distance.py --distance Zama amazing --alphabet StRiNg`: Compute the distance between
strings "Zama" and "amazing", considering the chars of "StRiNg" alphabet

```
Running distance between strings aa and ab for alphabet string:

    Computing Levenshtein between strings 'aa' and 'ab' - distance is 1, computed in 4.13 seconds

Successful end
```

FIXME: re-run when the semantic bug is found

- `levenshtein_distance.py --autotest`: Run random tests with the alphabet.

```
Making random tests with alphabet string
Letters are abcdefghijklmnopqrstuvwxyz

Computations in simulation

    Computing Levenshtein between strings '' and '' - OK
    Computing Levenshtein between strings '' and 'u' - OK
    Computing Levenshtein between strings '' and 'nh' - OK
    Computing Levenshtein between strings '' and 'fmf' - OK
    Computing Levenshtein between strings '' and 'cljm' - OK
    Computing Levenshtein between strings 'v' and '' - OK
    Computing Levenshtein between strings 'v' and 'a' - OK
    Computing Levenshtein between strings 'v' and 'hp' - OK
    Computing Levenshtein between strings 'g' and 'ktk' - OK
    Computing Levenshtein between strings 'o' and 'ydqu' - OK
    Computing Levenshtein between strings 'ke' and '' - OK
    Computing Levenshtein between strings 'eu' and 'w' - OK
    Computing Levenshtein between strings 'hi' and 'gz' - OK
    Computing Levenshtein between strings 'mx' and 'tbw' - OK
    Computing Levenshtein between strings 'uh' and 'lgad' - OK
    Computing Levenshtein between strings 'xpj' and '' - OK
    Computing Levenshtein between strings 'cdt' and 'f' - OK
    Computing Levenshtein between strings 'trl' and 'rl' - OK
    Computing Levenshtein between strings 'zai' and 'pqo' - OK
    Computing Levenshtein between strings 'vac' and 'nrov' - OK
    Computing Levenshtein between strings 'rnay' and '' - OK
    Computing Levenshtein between strings 'xnfg' and 'o' - OK
    Computing Levenshtein between strings 'jdgl' and 'ra' - OK
    Computing Levenshtein between strings 'wpyq' and 'jxp' - OK
    Computing Levenshtein between strings 'enpt' and 'hvfb' - OK

Computations in FHE

    Computing Levenshtein between strings '' and '' - OK in 0.01 seconds
    Computing Levenshtein between strings '' and 'u' - OK in 0.01 seconds
    Computing Levenshtein between strings '' and 'nh' - OK in 0.01 seconds
    Computing Levenshtein between strings '' and 'fmf' - OK in 0.01 seconds
    Computing Levenshtein between strings '' and 'cljm' - OK in 0.01 seconds
    Computing Levenshtein between strings 'v' and '' - OK in 0.01 seconds
    Computing Levenshtein between strings 'v' and 'a' - OK in 1.75 seconds
    Computing Levenshtein between strings 'v' and 'hp' - OK in 1.77 seconds
    Computing Levenshtein between strings 'g' and 'ktk' - OK in 2.78 seconds
    Computing Levenshtein between strings 'o' and 'ydqu' - OK in 3.61 seconds
    Computing Levenshtein between strings 'ke' and '' - OK in 0.01 seconds
    Computing Levenshtein between strings 'eu' and 'w' - OK in 1.73 seconds
    Computing Levenshtein between strings 'hi' and 'gz' - OK in 3.53 seconds
    Computing Levenshtein between strings 'mx' and 'tbw' - OK in 5.25 seconds
    Computing Levenshtein between strings 'uh' and 'lgad' - OK in 7.21 seconds
    Computing Levenshtein between strings 'xpj' and '' - OK in 0.01 seconds
    Computing Levenshtein between strings 'cdt' and 'f' - OK in 2.53 seconds
    Computing Levenshtein between strings 'trl' and 'rl' - OK in 5.32 seconds
    Computing Levenshtein between strings 'zai' and 'pqo' - OK in 7.93 seconds
    Computing Levenshtein between strings 'vac' and 'nrov' - OK in 10.73 seconds
    Computing Levenshtein between strings 'rnay' and '' - OK in 0.01 seconds
    Computing Levenshtein between strings 'xnfg' and 'o' - OK in 3.50 seconds
    Computing Levenshtein between strings 'jdgl' and 'ra' - OK in 7.01 seconds
    Computing Levenshtein between strings 'wpyq' and 'jxp' - OK in 10.67 seconds
    Computing Levenshtein between strings 'enpt' and 'hvfb' - OK in 14.30 seconds

Successful end
```

- `levenshtein_distance.py --autoperf`: Benchmark with random strings, for the different alphabets.

```

Typical performances for alphabet ACTG, with string of maximal length:

    Computing Levenshtein between strings 'GGAA' and 'AATT' - OK in 5.12 seconds
    Computing Levenshtein between strings 'TGCG' and 'ACAG' - OK in 5.00 seconds
    Computing Levenshtein between strings 'ATAC' and 'CTAA' - OK in 4.94 seconds

Typical performances for alphabet string, with string of maximal length:

    Computing Levenshtein between strings 'mtpp' and 'qujk' - OK in 15.48 seconds
    Computing Levenshtein between strings 'sucl' and 'teeu' - OK in 14.22 seconds
    Computing Levenshtein between strings 'prej' and 'latp' - OK in 14.07 seconds

Typical performances for alphabet STRING, with string of maximal length:

    Computing Levenshtein between strings 'ATRC' and 'VHCZ' - OK in 15.65 seconds
    Computing Levenshtein between strings 'BOPL' and 'AUVT' - OK in 14.38 seconds
    Computing Levenshtein between strings 'AMLK' and 'HEZX' - OK in 14.22 seconds

Typical performances for alphabet StRiNg, with string of maximal length:

    Computing Levenshtein between strings 'uIWB' and 'aYZR' - OK in 29.01 seconds
    Computing Levenshtein between strings 'adWI' and 'OXyg' - OK in 27.17 seconds
    Computing Levenshtein between strings 'jvhQ' and 'Weug' - OK in 26.55 seconds

Successful end

```

## Benchmarks on hpc7a

The benchmarks were done using Concrete 2.7 on `hpc7a` machine on AWS, and give:

```
Typical performances for alphabet ACTG, with string of maximal length:

    Computing Levenshtein between strings 'GGAA' and 'AATT' - OK in 5.12 seconds
    Computing Levenshtein between strings 'TGCG' and 'ACAG' - OK in 5.00 seconds
    Computing Levenshtein between strings 'ATAC' and 'CTAA' - OK in 4.94 seconds

Typical performances for alphabet string, with string of maximal length:

    Computing Levenshtein between strings 'mtpp' and 'qujk' - OK in 15.48 seconds
    Computing Levenshtein between strings 'sucl' and 'teeu' - OK in 14.22 seconds
    Computing Levenshtein between strings 'prej' and 'latp' - OK in 14.07 seconds

Typical performances for alphabet STRING, with string of maximal length:

    Computing Levenshtein between strings 'ATRC' and 'VHCZ' - OK in 15.65 seconds
    Computing Levenshtein between strings 'BOPL' and 'AUVT' - OK in 14.38 seconds
    Computing Levenshtein between strings 'AMLK' and 'HEZX' - OK in 14.22 seconds

Typical performances for alphabet StRiNg, with string of maximal length:

    Computing Levenshtein between strings 'uIWB' and 'aYZR' - OK in 29.01 seconds
    Computing Levenshtein between strings 'adWI' and 'OXyg' - OK in 27.17 seconds
    Computing Levenshtein between strings 'jvhQ' and 'Weug' - OK in 26.55 seconds

Successful end

```

FIXME: re-run the benchmarks on AWS





