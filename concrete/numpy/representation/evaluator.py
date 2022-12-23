"""
Declaration of various `Evaluator` classes, to make graphs picklable.
"""

# ruff: noqa: ARG002


class ConstantEvaluator:
    """
    ConstantEvaluator class, to evaluate Operation.Constant nodes.
    """

    def __init__(self, properties):
        self.properties = properties

    def __call__(self, *args, **kwargs):
        return self.properties["constant"]


class InputEvaluator:
    """
    InputEvaluator class, to evaluate Operation.Input nodes.
    """

    def __call__(self, *args, **kwargs):
        return args[0]


class GenericEvaluator:
    """
    GenericEvaluator class, to evaluate Operation.Generic nodes.
    """

    def __init__(self, operation, properties):
        self.operation = operation
        self.properties = properties

    def __call__(self, *args, **kwargs):
        return self.operation(*args, *self.properties["args"], **self.properties["kwargs"])


class GenericTupleEvaluator:
    """
    GenericEvaluator class, to evaluate Operation.Generic nodes where args are packed in a tuple.
    """

    def __init__(self, operation, properties):
        self.operation = operation
        self.properties = properties

    def __call__(self, *args, **kwargs):
        return self.operation(tuple(args), *self.properties["args"], **self.properties["kwargs"])
