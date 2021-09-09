# Debugging / Support / Submitting Issues

First, let's not forget that this version of Concrete framework is a beta product, meaning that it is not completely polished, contains several bugs (would they be known or unknown at this time). Also, let's not forget that FHE is a highly hot topic, and notably, that it cannot be considered as a solved problem.

Anyway, let's list some ways to debug your problems here. If nothing seems conclusive, you can still report the issue, as explained in a later section of this page.

## Is it a bug by the framework or by the user?

If ever your numpy program fails, it may be because:
- of bugs due to Concrete framework
- of bugs due to the user, notably who would have a bug without even considering FHE (does the function you want to compile run well with numpy?), or who would not use the framework as expected or not consider the limits of the framework.

For the latter kind of bugs, we encourage the user to have a look at:
- the error message she gets
- the documentation of the product
- the known limits of the product (such as the reduced set of supported operations at this time, or the limited precision of the computations).

Once the user has tried to see if the bug was not her own, it is time to go further.

## Having a reproducible bug

Once you're sure it is a bug, it would be nice to try to:
- make it highly reproducible: e.g., by reducing as much the randomness as possible; e.g., if you can find an input which fails, there is no reason to let the input random
- reduce it to the smallest possible bug: it is easier to investigate bugs which are small, so when you have an issue, please try to reduce to a smaller issue, notably with less lines of code, smaller parameters, less complex function to compile, faster scripts etc.

## Asking the community

We have created a Slack channel (TODO: LINK TO BE ADDED), such that you can directly ask the developpers and community about your issue.

Hopefully, it is just a misunderstanding or a small mistake on your side, that one can help you fix easily. And, the good point with your feedback is that, once we have heard the problem or misunderstanding, we can make the documentation even clearer (such as, completing the FAQ).

## Having a look to the compilation artifacts

When things are more complicated, or if you want to have a look by yourself, you may want to have a look to the compilation reports, which are called artifacts. This is as simple as described in [TODO: add the link to the tutorial about having artifacts].

This function will create a directory, containing notably:
[TODO: Umut to fix / complete the following information]
- bounds.txt: a file describing the expected ranges of data in the different steps of the computation
- cryptographic_parameters.txt: a file describing the different keys
- ir_nodes.txt: a file describing the different nodes in the intermediate representation (IR)
- optimizations_applied.txt: a file describing the different optimizations which were applied
- target_nodes.txt: a file describing the different nodes in the VM graph

Attaching the artifact with your issue or Slack message may help people to have a look at the core of the problem.
The more precise your bug, the more likely we can reproduce and fix

[TODO: Umut, is it still needed or do we already have some of those information in artifacts?]
In order to simplify our work and let us reproduce your bug easily, any information is useful. Notably, in addition to the python script, some information like:
- the OS version
- the python version
- the python packages you use
- the reproducibility rate you see on your side
- any insight you might have on the bug
- any workaround you have been able to find
may be useful to us. Don't remember, Concrete is a project where we are open to contribution, more information at Contributing (TODO: add a link).

## Submitting an issue

In case you have a bug, which is reproducible, that you have reduced to a small piece of code,  we have our issue tracker (TODO: LINK TO BE ADDED). Remember that a well-described short issue is an issue which is more likely to be studied and fixed. The more issues we receive, the better the product will be.
