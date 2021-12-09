# Debugging / Support / Submitting Issues

First, let's not forget that this version of **Concrete** is a first version of the product, meaning that it is not completely finished, contains several bugs (would they be known or unknown at this time), and will improve over time with the feedback from early users. Also, let's not forget that FHE is a highly hot topic, and notably, that it cannot be considered as a solved problem.

Anyway, let's list some ways to debug your problems here. If nothing seems conclusive, you can still report the issue, as explained in a later section of this page.

## Is it a bug by the framework or by the user?

If ever your numpy program fails, it may be because:
- of bugs due to **Concrete**
- of bugs due to the user, notably who would have a bug without even considering FHE (does the function you want to compile run well with numpy?), or who would not use the framework as expected or not consider the limits of the framework.

For the latter kind of bugs, we encourage the user to have a look at:
- the error message she gets
- the documentation of the product
- the known limits of the product (such as the reduced set of supported operations at this time, or the limited precision of the computations).

Once the user has tried to see if the bug was not her own, it is time to go further.

## Is the inputset sufficiently representative?

A bug may happen if ever the inputset, which is internally used by the compilation core to set bit widths of some intermediate data, is not sufficiently representative. Notably, if ever, with all the inputs in the inputset, it appears that an intermediate data can be represented an `n`-bit integer, but for a particular computation, this same intermediate data needs a bit more bits to be represented, the FHE execution for this computation will result in a wrong output (as typically in integer overflows in classical programs).

So, in general, when a bug appears, it may be a good idea to enlarge the inputset, and try to have random-looking inputs in this latter, following distribution of inputs used with the function.

## Having a reproducible bug

Once you're sure it is a bug, it would be nice to try to:
- make it highly reproducible: e.g., by reducing as much the randomness as possible; e.g., if you can find an input which fails, there is no reason to let the input random
- reduce it to the smallest possible bug: it is easier to investigate bugs which are small, so when you have an issue, please try to reduce to a smaller issue, notably with less lines of code, smaller parameters, less complex function to compile, faster scripts etc.

## Asking the community

You can directly ask the developers and community about your issue on our Discourse server (link on the right of the top menu).

Hopefully, it is just a misunderstanding or a small mistake on your side, that one can help you fix easily. And, the good point with your feedback is that, once we have heard the problem or misunderstanding, we can make the documentation even clearer (such as, completing the FAQ).

## Having a look to the compilation artifacts

When things are more complicated, or if you want to have a look by yourself, you may want to have a look to the compilation reports, which are called artifacts. This is as simple as described in [here](../tutorial/COMPILATION_ARTIFACTS.md)

The artifact system will create a directory, containing:
- **environment.txt:** information about your system
- **requirements.txt:** information about your python dependencies
- **function.txt:** source code of the function you are compiling
- **parameters.txt:** parameters you specified for compilation
- **1.initial.graph.txt:** textual representation of the initial computation graph right after tracing
- **1.initial.graph.png:** visual representation of the initial computation graph right after tracing
- ...
- **X.description.graph.txt:** textual representation of the Xth computation graph after topological transforms
- **X.description.graph.png:** visual representation of the Xth computation graph after topological transforms
- ...
- **N.final.graph.txt:** textual representation of the final computation graph right before MLIR conversion
- **N.final.graph.png:** visual representation of the final computation graph right before MLIR conversion
- **bounds.txt:** ranges of data in the different steps of the computation for the final graph that is being compiled
- **mlir.txt**: resulting MLIR code that is sent to the compiler (if compilation succeeded)
- **traceback.txt**: information about the error you encountered (if compilation failed)


Attaching the artifact with your issue or Slack message may help people to have a look at the core of the problem.
The more precise your bug, the more likely we can reproduce and fix

To simplify our work and let us reproduce your bug easily, we need all the information we can get. So, in addition to your python script, the following information would be very useful.
- compilation artifacts
- reproducibility rate you see on your side
- any insight you might have on the bug
- any workaround you have been able to find

Remember, **Concrete Framework** is a project where we are open to contributions, more information at [Contributing](../../dev/howto/CONTRIBUTING.md).

## Submitting an issue

In case you have a bug, which is reproducible, that you have reduced to a small piece of code,  we have our issue tracker (link on the right of the top menu). Remember that a well-described short issue is an issue which is more likely to be studied and fixed. The more issues we receive, the better the product will be.
